"""Groww fund page HTML parser — extract fund facts and section text blocks."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from src.constants import PARSED_DATA_DIR, RAW_DATA_DIR, SCHEME_NAMES, SCHEME_URLS
from src.ingestion.fetcher import MANIFEST_FILENAME, SchemeConfig, load_schemes

PARSED_MANIFEST_FILENAME = "parse_manifest.json"

logger = logging.getLogger(__name__)

# Headings whose following content is fund-relevant (chrome headings excluded).
DOM_SECTION_HEADINGS = (
    "Minimum investments",
    "Exit load, stamp duty and tax",
    "Fund management",
    "Investment Objective",
)

NOISE_HEADINGS = frozenset(
    {
        "Return calculator",
        "Returns and rankings",
        "Compare similar funds",
        "Holdings",
        "Understand terms",
        "Fund house",
    }
)


@dataclass
class TextBlock:
    section_title: str
    text: str


@dataclass
class ParsedPage:
    scheme_id: str
    text_blocks: list[TextBlock]
    nav_date: str | None
    fetched_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme_id": self.scheme_id,
            "text_blocks": [asdict(block) for block in self.text_blocks],
            "nav_date": self.nav_date,
            "fetched_at": self.fetched_at,
        }


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _format_inr(amount: float | int | None) -> str | None:
    if amount is None:
        return None
    if float(amount).is_integer():
        return f"₹{int(amount)}"
    return f"₹{amount}"


def _format_aum_crores(aum: float | None) -> str | None:
    if aum is None:
        return None
    return f"₹{aum:,.2f} Cr"


def extract_mf_server_side_data(soup: BeautifulSoup) -> dict[str, Any] | None:
    """Extract structured fund data embedded in Groww's __NEXT_DATA__ script."""
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        return None
    try:
        payload = json.loads(script.string)
        return payload["props"]["pageProps"]["mfServerSideData"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse __NEXT_DATA__: %s", exc)
        return None


def _build_text_blocks_from_json(mf: dict[str, Any]) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    scheme_name = mf.get("scheme_name") or mf.get("fund_name") or "Fund"

    nav = mf.get("nav")
    nav_date = mf.get("nav_date")
    expense_ratio = mf.get("expense_ratio")
    min_sip = mf.get("min_sip_investment")
    min_lumpsum = mf.get("min_investment_amount")
    aum = _format_aum_crores(mf.get("aum"))
    risk = mf.get("nfo_risk")
    category = mf.get("category")
    sub_category = mf.get("sub_category")
    plan_type = mf.get("plan_type")
    scheme_type = mf.get("scheme_type")

    key_facts: list[str] = [f"{scheme_name} key fund facts."]
    if nav is not None:
        key_facts.append(f"NAV: {nav}.")
    if nav_date:
        key_facts.append(f"NAV date: {nav_date}.")
    if expense_ratio is not None:
        key_facts.append(f"Expense ratio: {float(expense_ratio)}%.")
    if min_sip is not None:
        key_facts.append(f"Minimum SIP investment: {_format_inr(min_sip)}.")
    if min_lumpsum is not None:
        key_facts.append(f"Minimum lumpsum investment: {_format_inr(min_lumpsum)}.")
    if aum:
        key_facts.append(f"Fund size (AUM): {aum}.")
    if risk:
        key_facts.append(f"Riskometer: {risk}.")
    if category:
        line = f"Category: {category}"
        if sub_category:
            line += f" — {sub_category}"
        key_facts.append(f"{line}.")
    if plan_type and scheme_type:
        key_facts.append(f"Plan: {plan_type} {scheme_type}.")

    blocks.append(TextBlock(section_title="Key fund facts", text=" ".join(key_facts)))

    exit_load = mf.get("exit_load")
    if exit_load:
        blocks.append(TextBlock(section_title="Exit load", text=_normalize_whitespace(str(exit_load))))

    benchmark_name = mf.get("benchmark_name") or mf.get("benchmark")
    if benchmark_name:
        blocks.append(
            TextBlock(
                section_title="Benchmark",
                text=f"Fund benchmark: {_normalize_whitespace(str(benchmark_name))}.",
            )
        )

    description = mf.get("description")
    if description:
        blocks.append(
            TextBlock(
                section_title="Investment objective",
                text=_normalize_whitespace(str(description)),
            )
        )

    stamp_duty = mf.get("stamp_duty")
    tax_impact = (mf.get("category_info") or {}).get("tax_impact")
    tax_lines = []
    if stamp_duty:
        tax_lines.append(f"Stamp duty on investment: {stamp_duty}.")
    if tax_impact:
        tax_lines.append(_normalize_whitespace(str(tax_impact)))
    if tax_lines:
        blocks.append(TextBlock(section_title="Stamp duty and tax", text=" ".join(tax_lines)))

    manager_blocks = _manager_blocks_from_json(mf)
    blocks.extend(manager_blocks)

    return blocks


def _manager_blocks_from_json(mf: dict[str, Any]) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    details = mf.get("fund_manager_details") or []
    if details:
        lines: list[str] = []
        for manager in details:
            name = manager.get("person_name")
            if not name:
                continue
            parts = [name]
            if manager.get("education"):
                parts.append(f"Education: {_normalize_whitespace(str(manager['education']))}")
            if manager.get("experience"):
                parts.append(f"Experience: {_normalize_whitespace(str(manager['experience']))}")
            lines.append(" ".join(parts))
        if lines:
            blocks.append(TextBlock(section_title="Fund management", text=" ".join(lines)))
        return blocks

    fund_manager = mf.get("fund_manager")
    if fund_manager:
        blocks.append(
            TextBlock(
                section_title="Fund management",
                text=f"Fund manager: {_normalize_whitespace(str(fund_manager))}.",
            )
        )
    return blocks


def _find_heading(soup: BeautifulSoup, title: str) -> Tag | None:
    target = title.casefold()
    for tag in soup.find_all(["h2", "h3", "h4"]):
        text = tag.get_text(strip=True)
        if text.casefold() == target or target in text.casefold():
            return tag
    return None


def _extract_section_after_heading(heading: Tag, *, max_chars: int = 4000) -> str:
    """Collect text from siblings until the next major heading."""
    parts: list[str] = []
    for sibling in heading.next_siblings:
        if isinstance(sibling, Tag):
            if sibling.name in {"h2", "h3"}:
                break
            if sibling.get_text(strip=True):
                parts.append(sibling.get_text(" ", strip=True))
        elif isinstance(sibling, str) and sibling.strip():
            parts.append(sibling.strip())
    text = _normalize_whitespace(" ".join(parts))
    return text[:max_chars]


def _extract_dom_sections(soup: BeautifulSoup) -> list[TextBlock]:
    """Extract supplemental sections from visible HTML headings."""
    blocks: list[TextBlock] = []
    for title in DOM_SECTION_HEADINGS:
        heading = _find_heading(soup, title)
        if not heading:
            continue
        text = _extract_section_after_heading(heading)
        if len(text) < 20:
            continue
        blocks.append(TextBlock(section_title=title, text=text))

    about_heading = soup.find(
        "h3",
        class_=lambda c: c and any("investmentObjective_heading" in cls for cls in c),
    )
    if about_heading:
        container = about_heading.find_parent("div")
        if container:
            about_text = container.get_text(" ", strip=True)
            about_text = re.sub(r"^About\s*", "", about_text, flags=re.IGNORECASE)
            about_text = _normalize_whitespace(about_text)
            if len(about_text) >= 40 and "Investment Objective" not in about_text[:30]:
                blocks.append(TextBlock(section_title="About the fund", text=about_text[:2000]))

    return blocks


def _dedupe_text_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Keep the richest block per section title."""
    by_title: dict[str, TextBlock] = {}
    order: list[str] = []
    for block in blocks:
        key = block.section_title.casefold()
        existing = by_title.get(key)
        if existing is None:
            by_title[key] = block
            order.append(key)
        elif len(block.text) > len(existing.text):
            by_title[key] = block
    return [by_title[key] for key in order]


def _load_fetched_at(scheme_id: str, raw_dir: Path) -> str:
    manifest_path = raw_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        if scheme_id in manifest and manifest[scheme_id].get("fetched_at"):
            return manifest[scheme_id]["fetched_at"]

    html_path = raw_dir / f"{scheme_id}.html"
    if html_path.exists():
        ts = datetime.fromtimestamp(html_path.stat().st_mtime, tz=UTC)
        return ts.replace(microsecond=0).isoformat()

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def parse_html(
    html: str,
    scheme_id: str,
    *,
    fetched_at: str | None = None,
    raw_dir: Path | None = None,
) -> ParsedPage:
    """Parse Groww HTML into structured text blocks."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    blocks: list[TextBlock] = []
    mf = extract_mf_server_side_data(BeautifulSoup(html, "html.parser"))
    nav_date: str | None = None

    if mf:
        nav_date = mf.get("nav_date")
        blocks.extend(_build_text_blocks_from_json(mf))
    else:
        logger.warning("No mfServerSideData for %s — using DOM-only extraction", scheme_id)

    blocks.extend(_extract_dom_sections(soup))
    blocks = _dedupe_text_blocks(blocks)

    if not blocks:
        raise ValueError(f"No fund content extracted for scheme {scheme_id!r}")

    out_dir = raw_dir or RAW_DATA_DIR
    resolved_fetched_at = fetched_at or _load_fetched_at(scheme_id, out_dir)

    return ParsedPage(
        scheme_id=scheme_id,
        text_blocks=blocks,
        nav_date=nav_date,
        fetched_at=resolved_fetched_at,
    )


def parse_raw_file(
    scheme_id: str,
    *,
    raw_dir: Path | None = None,
) -> ParsedPage:
    """Parse a saved raw HTML snapshot for one scheme."""
    out_dir = raw_dir or RAW_DATA_DIR
    html_path = out_dir / f"{scheme_id}.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Raw HTML not found: {html_path}")

    html = html_path.read_text(encoding="utf-8")
    fetched_at = _load_fetched_at(scheme_id, out_dir)
    return parse_html(html, scheme_id, fetched_at=fetched_at, raw_dir=out_dir)


def parse_all(
    *,
    scheme_id: str | None = None,
    raw_dir: Path | None = None,
    schemes: list[SchemeConfig] | None = None,
) -> list[ParsedPage]:
    """Parse all (or one) saved raw HTML snapshots."""
    configured = schemes or load_schemes()
    if scheme_id is not None:
        configured = [s for s in configured if s.scheme_id == scheme_id]
        if not configured:
            known = {s.scheme_id for s in (schemes or load_schemes())}
            raise ValueError(f"Unknown scheme_id: {scheme_id!r}. Valid: {sorted(known)}")

    return [parse_raw_file(s.scheme_id, raw_dir=raw_dir) for s in configured]


def _page_to_review_dict(page: ParsedPage) -> dict[str, Any]:
    return {
        "scheme_id": page.scheme_id,
        "scheme_name": SCHEME_NAMES.get(page.scheme_id),
        "source_url": SCHEME_URLS.get(page.scheme_id),
        "nav_date": page.nav_date,
        "fetched_at": page.fetched_at,
        "text_block_count": len(page.text_blocks),
        "text_blocks": [asdict(block) for block in page.text_blocks],
    }


def save_parsed_page(
    page: ParsedPage,
    *,
    parsed_dir: Path | None = None,
) -> Path:
    """Write one scheme's parsed output to data/parsed/{scheme_id}.json."""
    out_dir = parsed_dir or PARSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{page.scheme_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_page_to_review_dict(page), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def save_parsed_pages(
    pages: list[ParsedPage],
    *,
    parsed_dir: Path | None = None,
) -> list[Path]:
    """Write parsed outputs and a summary manifest for review."""
    out_dir = parsed_dir or PARSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [save_parsed_page(page, parsed_dir=out_dir) for page in pages]
    manifest = {
        "parsed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "scheme_count": len(pages),
        "schemes": [
            {
                "scheme_id": page.scheme_id,
                "scheme_name": SCHEME_NAMES.get(page.scheme_id),
                "file": f"{page.scheme_id}.json",
                "text_block_count": len(page.text_blocks),
                "nav_date": page.nav_date,
                "fetched_at": page.fetched_at,
            }
            for page in pages
        ],
    }
    manifest_path = out_dir / PARSED_MANIFEST_FILENAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info("Saved %s parsed file(s) to %s", len(paths), out_dir)
    return paths
