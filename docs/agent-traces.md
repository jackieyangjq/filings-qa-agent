# Agent traces

`filings-qa agent` answers a question by letting Gemini call three tools, then write an answer that gives a source for
every fact. Below are three compound questions run on 2026-09-23 with `gemini-3.5-flash-lite` (free tier) over the 48
filings of `companies.yaml`: 12 companies, each with its latest 10-K and three latest 10-Qs. For each question: every
tool call in order, with its arguments and a one-line summary of what came back; the usage; the answer as the model
wrote it; and what a check of the answer against the sources found. The full JSON traces are written to `data/traces/`,
which is not part of the repository.

| Tool | Returns | Source |
|---|---|---|
| `search_filings(query, ticker, form, filed, k)` | passages (chunks) of the filings, each with its chunk id, company, form, filing date, period, section and text | local SQLite full-text and vector index, hybrid search |
| `get_price(ticker, start, end or trading_days)` | daily closes, each with a count of trading days from the first | Yahoo Finance through yfinance, adjusted for splits and dividends |
| `get_news(ticker, days, topic)` | up to 10 headlines with date, source, title and link | Google News RSS search, no key needed (with `FINNHUB_API_KEY` set, Finnhub's company news instead, an option added after these runs) |

How a run works:

- The model gets the question, today's date, the rules (use only what the tools return; put a source after every
  fact: a chunk id for a filing passage, the tool and dates for a price or a headline; no advice to buy, sell or
  hold) and the tool declarations. The description of `search_filings` lists the stored filings with their dates.
- A reply may ask for several tool calls at once: one round. Every call in it runs and its result goes back to the
  model. After `--max-steps` rounds (default 6) further calls are refused and the model must answer without tools;
  the result is then marked truncated.
- A call with a missing or unknown argument is not run: the model gets the error and can fix the call in its next
  round (question 1 shows this).
- After the answer, wording that reads as investment advice gets a disclaimer, and chunk ids that no search of the run
  returned are listed as unsupported.

To run one: `filings-qa agent "<question>"`, after `pip install -e '.[llm,embed,tools]'`, `filings-qa ingest` (with
`SEC_USER_AGENT` set), `filings-qa index` and setting `GEMINI_API_KEY`.

## 1. NVIDIA: data center demand in the last three filings, and the stock after each

**Question:** How has NVIDIA described data center demand across its last three filings, and how did the stock move in
the five trading days after each filing date?

| # | Round | Tool | Arguments | What came back | Time |
|---|---|---|---|---|---|
| 1 | 1 | `search_filings` | `ticker="NVDA", form="10-Q"` | error: ToolError: search_filings: missing query (it takes query, ticker, form, k, filed) | 0.00 s |
| 2 | 1 | `search_filings` | `query="data center demand revenue customers", ticker="NVDA", form="10-K", filed="2026-02-25"` | 6 passages: NVDA-10-K-20260225-15-071, NVDA-10-K-20260225-7-002, NVDA-10-K-20260225-1A-012, NVDA-10-K-20260225-1A-015, NVDA-10-K-20260225-1A-010, NVDA-10-K-20260225-1A-013 | 0.31 s |
| 3 | 1 | `get_price` | `ticker="NVDA", start="2026-02-25", trading_days=5` | 6 closes: 2026-02-25 195.10 to 2026-03-04 182.61 (-6.4%) | 0.58 s |
| 4 | 1 | `get_price` | `ticker="NVDA", start="2026-05-20", trading_days=5` | 6 closes: 2026-05-20 222.96 to 2026-05-28 213.76 (-4.1%) | 0.06 s |
| 5 | 1 | `get_price` | `ticker="NVDA", start="2026-08-26", trading_days=5` | 6 closes: 2026-08-26 209.43 to 2026-09-02 224.16 (+7.0%) | 0.06 s |
| 6 | 2 | `search_filings` | `query="data center demand revenue customers", ticker="NVDA", form="10-Q", filed="2026-05-20"` | 6 passages: NVDA-10-Q-20260520-2-009, NVDA-10-Q-20260520-1A-006, NVDA-10-Q-20260520-1A-005, NVDA-10-Q-20260520-1A-009, NVDA-10-Q-20260520-1A-008, NVDA-10-Q-20260520-1A-007 | 0.03 s |
| 7 | 2 | `search_filings` | `query="data center demand revenue customers", ticker="NVDA", form="10-Q", filed="2026-08-26"` | 6 passages: NVDA-10-Q-20260826-2-005, NVDA-10-Q-20260826-1A-002, NVDA-10-Q-20260826-2-007, NVDA-10-Q-20260826-2-006, NVDA-10-Q-20260826-1A-003, NVDA-10-Q-20260826-2-010 | 0.02 s |

**Usage:** 3 model calls (2 rounds of tool calls, then the answer), 7 tool calls, 17,594 input and 1,287 output tokens, 6.4 s. Truncated: no. Advice wording: none. Chunk ids cited that no search returned: none.

> NVIDIA has described data center demand and related infrastructure dynamics across its last three SEC filings as follows:
>
> * **Annual Report (Form 10-K for the fiscal year ended January 25, 2026, filed February 25, 2026):** NVIDIA reported that revenue growth was driven by data center compute and networking platforms for accelerated computing and AI solutions, with Blackwell architectures representing the majority of Data Center revenue [NVDA-10-K-20260225-7-002]. It emphasized that the availability of data centers, energy, and capital to support customer and partner buildouts was crucial, noting that shortages, regulatory/construction challenges in expanding energy capacity, or constrained access to capital could delay deployments or reduce the scale of adoption [NVDA-10-K-20260225-7-002].
> * **Quarterly Report (Form 10-Q for the quarterly period ended April 26, 2026, filed May 20, 2026):** NVIDIA announced record Data Center revenue of $75.2 billion (up 92% year-over-year and 21% sequentially), driven by the ramp of Blackwell 300 products and demand for InfiniBand, Spectrum-X Ethernet, and NVLink solutions [NVDA-10-Q-20260520-2-009]. Hyperscalers remained at approximately 50% of Data Center revenue, with the remaining 50% coming from continued customer diversification (including AI clouds, industrial, enterprise, and sovereign customers) [NVDA-10-Q-20260520-2-009]. The company continued to highlight potential bottlenecks regarding data center capacity, energy, and capital constraints [NVDA-10-Q-20260520-1A-007].
> * **Quarterly Report (Form 10-Q for the quarterly period ended July 26, 2026, filed August 26, 2026):** NVIDIA noted that it significantly increased its supply and capacity commitments from $119 billion in the previous quarter to $279 billion to meet future demand [NVDA-10-Q-20260826-2-005]. It introduced new business models and infrastructure initiatives with select AI cloud partners to address constraints faced by AI clouds and model makers—such as shortages of land, power, shell, and investment-grade financing capacity—by undertaking select data center infrastructure commitments, leases, and financial guarantees totaling tens of billions of dollars [NVDA-10-Q-20260826-1A-003, NVDA-10-Q-20260826-2-006, NVDA-10-Q-20260826-2-010].
>
> **Stock Price Movements in the Five Trading Days After Each Filing Date**
>
> * **Following the February 25, 2026 10-K filing:** 
>   The stock moved from a close of **$195.10** on February 25, 2026, to a close of **$182.61** on March 4, 2026, down **6.4%** [get_price NVDA 2026-02-25 to trading_days 5].
> * **Following the May 20, 2026 10-Q filing:** 
>   The stock moved from a close of **$222.96** on May 20, 2026, to a close of **$213.76** on May 28, 2026, down **4.1%** [get_price NVDA 2026-05-20 to trading_days 5].
> * **Following the August 26, 2026 10-Q filing:** 
>   The stock moved from a close of **$209.43** on August 26, 2026, to a close of **$224.16** on September 2, 2026, up **7.0%** [get_price NVDA 2026-08-26 to trading_days 5].

**Checked against the sources:**

- The filings are the right three: NVIDIA's latest are the 10-K filed 2026-02-25 and the 10-Qs filed 2026-05-20 and
  2026-08-26 (the 10-Q of 2025-11-19 is older). The model took them from the filing list in the search tool's
  description and searched each one by its date.
- Every cited passage says what the answer attributes to it, with two loose spots: the passage gives Data Center
  revenue of $75.2 billion, up 92%, without calling it a record; and "tens of billions of dollars" stands for the
  $36 billion of AI cloud commitments in NVDA-10-Q-20260826-2-010.
- The fifth trading day after each filing is right: 2026-03-04; 2026-05-28 (markets were closed on Memorial Day,
  2026-05-25); 2026-09-02. The percentages follow from the closes.
- Missed: the latest quarter's headline figure, Data Center revenue of $89.0 billion, up 117% (passage
  NVDA-10-Q-20260826-2-015), was not among the six passages returned for that filing, so the answer describes the
  latest 10-Q through its supply commitments and infrastructure constraints instead.

## 2. Apple and Microsoft: capital expenditure plans, and related news

**Question:** Compare what Apple and Microsoft said about capital expenditure plans in their most recent 10-K, and list
any related news from the past week.

| # | Round | Tool | Arguments | What came back | Time |
|---|---|---|---|---|---|
| 1 | 1 | `search_filings` | `query="capital expenditures plans property plant and equipment additions", ticker="AAPL", form="10-K", k=5` | 5 passages: AAPL-10-K-20251031-8-020, AAPL-10-K-20251031-8-019, AAPL-10-K-20251031-8-005, AAPL-10-K-20251031-8-008, AAPL-10-K-20251031-8-025 | 0.29 s |
| 2 | 1 | `search_filings` | `query="capital expenditures plans property plant and equipment additions", ticker="MSFT", form="10-K", filed="2026-07-29", k=5` | 5 passages: MSFT-10-K-20260729-7-029, MSFT-10-K-20260729-8-026, MSFT-10-K-20260729-8-039, MSFT-10-K-20260729-8-054, MSFT-10-K-20260729-7-025 | 0.01 s |
| 3 | 1 | `get_news` | `ticker="AAPL", topic="capital expenditure plans", days=7` | no headlines in this period | 0.35 s |
| 4 | 1 | `get_news` | `ticker="MSFT", topic="capital expenditure plans", days=7` | 1 headline, 2026-09-22 (Stocktwits): on Morgan Stanley's view that the AI capital spending of Amazon, Alphabet and Microsoft will pay off *(title paraphrased)* | 0.38 s |

**Usage:** 2 model calls (1 round of tool calls, then the answer), 4 tool calls, 10,917 input and 658 output tokens, 4.5 s. Truncated: no. Advice wording: none. Chunk ids cited that no search returned: none.

> Based on Apple's most recent Form 10-K (for the fiscal year ended September 27, 2025) and Microsoft's most recent Form 10-K (for the fiscal year ended June 30, 2026), here is a comparison of what they stated regarding capital expenditure plans:
>
> **Apple Inc.**
> * **Capital Expenditure Plans:** Apple’s 2025 Form 10-K does not explicitly discuss qualitative forward-looking plans or projections for capital expenditures. However, its cash flow statement reports historical payments for the acquisition of property, plant, and equipment of $12,715 million for fiscal year 2025 (up from $9,447 million in 2024 and $10,959 million in 2023) [AAPL-10-K-20251031-8-005].
>
> **Microsoft Corporation**
> * **Capital Expenditure Plans:** Microsoft provided qualitative details indicating that additions to property and equipment will continue, driven by new facilities, datacenters, and computer systems for R&D, sales, marketing, support, and administrative staff [MSFT-10-K-20260729-7-029]. Specifically, Microsoft stated it will continue investing in capital expenditures to support growth in cloud offerings and investments in AI training and other infrastructure [MSFT-10-K-20260729-7-029]. Additionally, Microsoft noted that as of June 30, 2026, it had committed $34.6 billion for the construction of new buildings, building improvements, and leasehold improvements, primarily related to datacenters [MSFT-10-K-20260729-8-039].
>
> **Related News from the Past Week**
> * **Apple:** No news headlines specifically regarding capital expenditure plans were returned for Apple over the past week.
> * **Microsoft:** Stocktwits published an article titled *[title paraphrased here: why Morgan Stanley says the AI capital spending plans of Amazon, Alphabet and Microsoft will pay off]* on September 22, 2026 [get_news Stocktwits 2026-09-22].

**Checked against the sources:**

- The 10-Ks are the most recent ones: Apple's filed 2025-10-31 (fiscal year ended 2025-09-27) and Microsoft's filed
  2026-07-29 (fiscal year ended 2026-06-30).
- The cited passages say what the answer says: Apple's cash flow statement lists payments for property, plant and
  equipment of $12,715, $9,447 and $10,959 million for 2025, 2024 and 2023; Microsoft's "Other planned uses of
  capital" and its note on property and equipment give the plans and the $34.6 billion committed for construction,
  mainly of datacenters.
- "Apple's 2025 Form 10-K does not explicitly discuss ... forward-looking plans" is the model's conclusion from the
  five passages it retrieved, not something a passage states.
- News: with the topic "capital expenditure plans", the Google News search had no Apple headline from the past week
  and one for Microsoft.

## 3. Tesla: tariff risks, and the stock over the past month

**Question:** What risks related to tariffs does Tesla disclose, and what has the stock done over the past month?

| # | Round | Tool | Arguments | What came back | Time |
|---|---|---|---|---|---|
| 1 | 1 | `search_filings` | `query="tariffs trade duties import restrictions", ticker="TSLA", form="10-Q"` | 6 passages: TSLA-10-Q-20260723-2-009, TSLA-10-Q-20251023-2-009, TSLA-10-Q-20260423-2-009, TSLA-10-Q-20260423-1-041, TSLA-10-Q-20260723-1-048, TSLA-10-Q-20251023-2-006 | 0.28 s |
| 2 | 1 | `get_price` | `ticker="TSLA", start="2026-08-23", trading_days=22` | 22 closes: 2026-08-24 348.95 to 2026-09-23 379.86 (+8.9%); the last is intraday, not a close | 0.55 s |

**Usage:** 2 model calls (1 round of tool calls, then the answer), 2 tool calls, 7,205 input and 553 output tokens, 4.0 s. Truncated: no. Advice wording: none. Chunk ids cited that no search returned: none.

> Based on Tesla's SEC filings and stock price history, here are the tariff-related disclosures and the stock's performance over the past month:
>
> **Tariff-Related Risks Disclosed by Tesla**
> * **Impact of Trade Policy and Tariffs on Costs and Demand:** Tesla notes that changes in government and economic policies, incentives, or tariffs can impact its production, cost structure, and competitive landscape. Specifically, import tariffs and subsequent retaliatory tariffs could affect consumer spending and the demand for durable goods and related services [TSLA-10-Q-20260723-2-009].
> * **Global Supply Chain:** Changes to fiscal and trade policies regarding tariffs, export controls, and other restrictions can affect Tesla's global supply chain cost structure and component availability, impacting vehicle production and facility expansions [TSLA-10-Q-20251023-2-006].
> * **Potential Tariff Refunds (IEEPA Ruling):** Following a February 2026 U.S. Supreme Court ruling invalidating certain tariffs imposed under the International Emergency Economic Powers Act (IEEPA), Tesla disclosed that it may be eligible for refunds on previously paid import tariffs. However, because the recoverability and timing remain uncertain, Tesla has not recognized any receivable or expense offset. Tesla also noted that potential refunds could require contractual reductions in revenue for certain energy storage customers [TSLA-10-Q-20260723-1-048].
>
> **Stock Performance Over the Past Month**
> Over the past month (spanning 21 trading days from August 24, 2026, to September 22, 2026), Tesla's stock moved from a close of $348.95 on August 24, 2026, to $378.90 on September 22, 2026, representing an increase of approximately 8.6% [get_price TSLA 2026-08-23 to 2026-09-23]. *(Note: As of midday on September 23, 2026, the latest trading price was $379.86).*

**Checked against the sources:**

- The run was made at 15:36 in New York, with the market open. The tool marked the 2026-09-23 row as the latest price,
  not a close, and the answer treats it that way: its move ends at the 2026-09-22 close (378.90, confirmed by a fresh
  download), from 348.95 on 2026-08-24, up 8.6%. It calls the time "midday", which the tools did not say.
- For "the past month" the model asked for 22 trading days from 2026-08-23, a Sunday; the tool started at the next
  trading day, 2026-08-24, and stopped at today because later days have not happened yet.
- The three cited passages say what the answer says, including the note on possible refunds after the February 2026
  Supreme Court ruling on tariffs imposed under IEEPA.
- Missed: the model searched only the 10-Qs, so the risk factors of the 10-K filed 2026-01-29 are absent, e.g. that
  U.S. tariffs have raised Tesla's costs for vehicles built in and exported from the United States, and that some
  parts are difficult or impossible to source within the United States (TSLA-10-K-20260129-1A-022). An earlier run
  that searched the 10-K found them.

## What the first runs changed

The tools and the prompt were changed after the first real runs on the same day; each change answers a failure seen in
a trace.

1. **Wrong filings.** Asked for "the last three filings", the model could not know which filings exist; it described
   the 10-Qs of 2025-11-19 and 2026-05-20 and the 10-K, and missed the latest 10-Q (2026-08-26). Now the description of
   `search_filings` lists the stored filings with their dates, and `filed` restricts a search to one filing.
2. **Trading days counted as calendar days.** The "fifth trading day after the filing" came out anywhere from the
   fourth to the seventh (2026-03-06, the seventh, for a filing of 2026-02-25, for example). Now `get_price` takes
   `trading_days`: it returns the close on the start date and on the N trading days after it, so the move is the last
   close against the first.
3. **One bad round used up the budget.** The model sometimes leaves out a required argument in calls it sends
   together. While the limit counted single tool calls, two such calls left too few for the rest and the answer was
   cut short. Now the limit counts rounds, so a rejected call costs one retry.
4. **Half an answer.** Once the model wrote most of its answer in a reply that also asked for one more price, and its
   final reply only added the rest. Now the answer joins the text of all replies.
5. **An intraday price called a close.** During trading hours the model reported today's price as the day's close.
   Now that row carries a note saying it is the latest price, not a close (question 3 shows the result).
6. **News.** The ten newest headlines were all from one day and none was about capital expenditure. Now the tool
   keeps Google's top-ranked headlines of the period, newest first, and can add a topic to the search.

## Limits

- `gemini-3.5-flash-lite` does not give the same plan twice: one run of a question can search each filing, the next
  only some of them, and slips such as a missing argument come and go. Each trace above is the first run of its
  question after the last change, not the best of several.
- The agent can only cite what search returns, and search sometimes misses the most specific passage (question 1).
- Prices come from Yahoo Finance through yfinance, an unofficial interface. Headlines are what the outlets wrote,
  unchecked; their titles are paraphrased in this document.
