# Problem Statement: Mutual Fund FAQ Assistant (Facts-Only Q&A)

## Overview

The objective of this project is to build a **facts-only FAQ assistant** for mutual fund schemes, using **Groww** as the reference product context. The initial scope covers **five HDFC Mutual Fund schemes** across equity (large-cap, mid-cap, small-cap) and commodities (gold, silver). The assistant will answer objective, verifiable queries related to these schemes by retrieving information exclusively from official public sources, such as AMC (Asset Management Company) websites, AMFI, and SEBI.

The system must strictly avoid providing investment advice, opinions, or recommendations. Every response must include a single, clear source link and adhere to defined constraints around clarity, accuracy, and compliance.

## Objective

Design and implement a lightweight **Retrieval-Augmented Generation (RAG)**-based assistant that:

- Answers factual queries about mutual fund schemes
- Uses a curated corpus of official documents
- Provides concise, source-backed responses

## Target Users

- Retail investors comparing mutual fund schemes
- Customer support and content teams handling repetitive mutual fund queries

## Selected AMC & Schemes

**AMC:** [HDFC Mutual Fund](http://www.hdfcfund.com) (HDFC Asset Management Company Limited)

The RAG chatbot is scoped to the following **five schemes**. Groww fund pages are used as the **reference product context** only; the corpus must be built from official AMC, AMFI, and SEBI sources.

| Scheme | Category | Risk | Min. SIP | Groww reference |
| --- | --- | --- | --- | --- |
| HDFC Large Cap Fund Direct Growth | Equity — Large Cap | Very High | ₹100 | [View on Groww](https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth) |
| HDFC Mid Cap Fund Direct Growth | Equity — Mid Cap | Very High | ₹100 | [View on Groww](https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth) |
| HDFC Small Cap Fund Direct Growth | Equity — Small Cap | Very High | ₹100 | [View on Groww](https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth) |
| HDFC Gold ETF Fund of Fund Direct Plan Growth | Commodities — Gold | High | ₹100 | [View on Groww](https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth) |
| HDFC Silver ETF FoF Direct Growth | Commodities — Silver | Very High | ₹100 | [View on Groww](https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth) |

**Category coverage:** large-cap, mid-cap, small-cap, gold, and silver — spanning equity and commodity fund types.

## Scope of Work

### 1. Corpus Definition

**Status:** Reference and official corpus URLs identified (22 URLs).

#### Reference URLs (Groww — product context)

These URLs define the five in-scope schemes and serve as the Groww product reference. They are **not** authoritative retrieval sources for answers.

| Scheme | URL |
| --- | --- |
| HDFC Large Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| HDFC Mid Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth |
| HDFC Small Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| HDFC Gold ETF Fund of Fund Direct Plan Growth | https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth |
| HDFC Silver ETF FoF Direct Growth | https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth |

#### Official corpus URLs (authoritative sources)

**AMC scheme pages (Direct plan)**

| Scheme | URL |
| --- | --- |
| HDFC Large Cap Fund | https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct |
| HDFC Mid Cap Fund | https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct |
| HDFC Small Cap Fund | https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct |
| HDFC Gold ETF Fund of Fund | https://www.hdfcfund.com/explore/mutual-funds/hdfc-gold-etf-fund-fund/direct |
| HDFC Silver ETF Fund of Fund | https://www.hdfcfund.com/explore/mutual-funds/hdfc-silver-etf-fund-fund/direct |

Each scheme page links to scheme-specific **SID**, **KIM**, and **Fund Facts** downloads (e.g., dated November 21, 2025).

**Scheme documents (PDFs)**

| Document | URL |
| --- | --- |
| SID — HDFC Gold ETF Fund of Fund (Nov 21, 2025) | https://files.hdfcfund.com/s3fs-public/SID/2025-11/SID%20-%20HDFC%20Gold%20ETF%20Fund%20of%20Fund%20dated%20November%2021%2C%202025.pdf |
| KIM — HDFC Gold ETF Fund of Fund (Nov 21, 2025) | https://files.hdfcfund.com/s3fs-public/KIM/2025-11/KIM%20-%20HDFC%20Gold%20ETF%20Fund%20of%20Fund%20dated%20November%2021%2C%202025.pdf |
| Leaflet — HDFC Silver ETF & Silver ETF FoF (Jul 2025) | https://files.hdfcfund.com/s3fs-public/Others/2025-07/HDFC%20Silver%20ETF%20and%20HDFC%20Silver%20ETF%20Fund%20of%20Fund%20Leaflet%20%28July%202025%29.pdf |

Additional per-scheme SID/KIM PDFs for the equity funds are available via the AMC document hubs below.

**AMC shared resources**

| Resource | URL |
| --- | --- |
| HDFC Mutual Fund — home | https://www.hdfcfund.com |
| Scheme Information Documents (SID) hub | https://www.hdfcfund.com/mutual-funds/fund-documents/sid |
| Key Information Memorandum (KIM) hub | https://www.hdfcfund.com/mutual-funds/fund-documents/kim |
| Statutory disclosures | https://www.hdfcfund.com/statutory-disclosure |
| Investor FAQs (Direct plan) | https://www.hdfcfund.com/services/faqs/introduction-direct-plan |
| Gold & silver investment guide | https://www.hdfcfund.com/learners-corner/digital-gold-silver |
| Request account statement | https://www.hdfcfund.com/investor-services/request-statement |

**Regulatory / industry guidance**

| Resource | URL |
| --- | --- |
| AMFI — investor corner | https://www.amfiindia.com/investor-corner |
| SEBI — mutual funds (investor information) | https://investor.sebi.gov.in/mutualfund.html |

**Corpus indexing notes**

- Prioritize official AMC pages and PDFs for factual answers; cite the specific document or scheme page used.
- Groww URLs are for scheme selection and UI reference only.
- Refresh corpus when AMC publishes updated SID/KIM or factsheet revisions.

### 2. FAQ Assistant Requirements

The assistant must:

**Answer facts-only queries** about the selected HDFC schemes, such as:

- Expense ratio of a scheme (e.g., HDFC Mid Cap Fund Direct Growth)
- Exit load details (e.g., 1% if redeemed within 15 days for HDFC Gold ETF FoF)
- Minimum SIP amount
- Riskometer classification
- Benchmark index (e.g., NIFTY 100 Total Return Index for HDFC Large Cap Fund)
- Process to download statements or capital gains reports

**Ensure:**

- Each response is limited to a maximum of **3 sentences**
- Each response includes **exactly one citation link**
- Each response includes a footer:
  > Last updated from sources: `<date>`

### 3. Refusal Handling

The assistant must refuse non-factual or advisory queries, such as:

- "Should I invest in this fund?"
- "Which fund is better?"

Refusal responses should:

- Be polite and clearly worded
- Reinforce the facts-only limitation
- Provide a relevant educational link (e.g., AMFI or SEBI resource)

### 4. User Interface (Minimal)

The solution should include a simple interface with:

- A welcome message scoped to the five HDFC schemes above
- Three example questions, such as:
  - What is the expense ratio of HDFC Large Cap Fund Direct Growth?
  - What is the exit load for HDFC Gold ETF Fund of Fund?
  - What is the minimum SIP for HDFC Small Cap Fund Direct Growth?
- A visible disclaimer:
  > Facts-only. No investment advice.

## Constraints

### Data and Sources

- Use only official public sources (AMC, AMFI, SEBI)
- Do not use third-party blogs or aggregator websites
- Groww fund pages are reference context for UX and scheme selection only; they are **not** authoritative corpus sources

### Privacy and Security

Do not collect, store, or process:

- PAN or Aadhaar numbers
- Account numbers
- OTPs
- Email addresses or phone numbers

### Content Restrictions

- No investment advice or recommendations
- No performance comparisons or return calculations
- For performance-related queries, provide a link to the official factsheet only

### Transparency

- Responses must be short, factual, and verifiable
- Every answer must include a source link and last updated date

## Expected Deliverables

### README Document

- Setup instructions
- Selected AMC (HDFC Mutual Fund) and the five schemes listed above
- Architecture overview (RAG approach)
- Known limitations

### Disclaimer Snippet

> Facts-only. No investment advice.

## Success Criteria

- Accurate retrieval of factual mutual fund information
- Strict adherence to facts-only responses
- Consistent inclusion of valid source citations
- Proper refusal of advisory queries
- Clean, minimal, and user-friendly interface

## Summary

The goal is to build a trustworthy, transparent, and compliant mutual fund FAQ assistant for the five selected HDFC schemes that prioritizes **accuracy over intelligence**. The system should ensure that users receive only verified, source-backed financial information, without any advisory bias or speculative content.
