# Customer Intelligence & Support Copilot (Instamart, synthetic data)

## Overview

One repository, four connected parts, all on synthetic data (nothing here is real Instamart data). All money is in INR.

| Part | What it does | Main outputs |
|---|---|---|
| A | Audit and clean raw exports, store in SQLite, SQL + pandas analysis | `data/clean/*.csv`, `data/instamart.db`, `sql/queries.sql` |
| B | RFM + K-means segments, PCA, lapse prediction | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C | Ticket triage with privacy redaction, retries, validation, token budget, routing | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D | Policy RAG assistant and a tool-calling refund agent | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

`generate_data.py` and `mock_llm.py` are the supplied files, unmodified. Libraries: standard library, pandas, numpy, scikit-learn, joblib. No network, no API key. Every script must be run from the repository root.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py              # last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164
python src/a1_audit.py               # Part A audit
python src/a2_clean.py               # Part A cleaning -> data/clean/*.csv + data/instamart.db
python src/a3_validate.py            # prints ALL CHECKPOINTS PASSED
python src/a4_sql_pandas.py          # seven SQL queries + pandas cross-check + EDA
python src/b1_segments.py            # Part B RFM, K-means, PCA -> outputs/customer_segments.csv
python src/b2_lapse.py               # Part B lapse model -> lapse_model.joblib, retention_list.csv
python src/c_triage.py               # Part C -> triage_results.csv, triage_budget_demo.csv
python src/d_rag.py                  # Part D RAG chunking, retrieval, evaluation
python src/d_agent.py                # Part D agent -> agent_runs.jsonl, refund_ledger.jsonl, state/
```

Scripts are in `src/`. Part D's agent run deletes and rewrites `outputs/agent_runs.jsonl`, `outputs/agent_demo_runs.jsonl`, `outputs/refund_ledger.jsonl` and `state/`, so it can be re-run safely.

## Part A: Data Foundation

**Audit (raw tables, `src/a1_audit.py`).** Shapes: customers (1200, 5), products (40, 4), orders (13084, 9). Missing values (count / %): customers `age` 61 / 5.08%, `gender` 30 / 2.50%; orders `order_amount` 53 / 0.41% (the text `N/A` is read as missing by pandas; the other 67 bad entries are the text `error`), `delivery_minutes` 2541 / 19.42%; no other column has missing values. Data types: ids, city, dates and gender are text; age, quantity, discount, price are numeric; `order_amount` is text because of the planted bad entries.

| Audit figure | Value |
|---|---|
| raw order rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age | 61 |
| age outside 10-100 | 6 |
| distinct raw city spellings | 24 |
| distinct non-blank raw gender spellings | 10 |
| blank gender | 30 |

**Customer cleaning rules (`src/a2_clean.py`).**
1. City: strip spaces + title case, 24 spellings became 6.
2. Gender: mapped `m/male` to male, `f/female` to female, blank to unknown (610 / 560 / 30).
3. Age: 6 out-of-range ages (280, -4, 0, 150, 999, 200) set to missing, giving 67 missing; all 67 filled with the city median using `groupby().transform("median")`. Medians that were not whole numbers were rounded to whole years. Result: 0 missing, all ages 18-60.
4. `signup_date` converted to datetime.
Cities after cleaning: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.

**Order cleaning rules, applied in this order.**
1. `order_ts` converted to datetime (all 13,084 rows).
2. Exact duplicates removed: **119** rows.
3. Per `order_id` keep only the record with the latest `order_ts`: **656** earlier pending records dropped, leaving 12,309 rows.
4. `order_amount` converted to numeric (bad entries become missing); **112** missing amounts (the 120 raw bad entries minus those in rows already removed) refilled with `round(price x quantity x (1 - discount/100), 2)`.
5. Negative `delivery_minutes` set to missing (**105** values remained after de-duplication); for completed orders **774** missing delivery times (planted blanks + the negatives) filled with the median of the product category among completed orders (`groupby().transform`). Cancelled orders keep a missing delivery time because they were never delivered.

Result: 12,309 rows; 11,110 completed, 1,199 cancelled, 0 pending; 0 missing `order_amount`; 0 missing delivery among completed, 1,199 among cancelled; completed revenue INR 2,998,668.80.

**Store and validate.** Clean CSVs are in `data/clean/`; SQLite `data/instamart.db` has tables `customers`, `products`, `orders` (`order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `src/a3_validate.py` asserts 13 checkpoints and prints `ALL CHECKPOINTS PASSED`.

**SQL results** (queries in `sql/queries.sql`, executed in `src/a4_sql_pandas.py` with `pd.read_sql_query`):

| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTEs | 550 customers above the average spend of INR 2,552.06 |
| Q6 window ranking | top per city: Bengaluru C0082 12,856.20; Chennai C0930 10,151.40; Delhi C0291 13,742.60; Hyderabad C0743 12,730.05; Mumbai C0693 14,766.55; Pune C0426 10,377.10; 8 rows where RANK() differs from ROW_NUMBER() |
| Q7 running total | 183 daily rows; last cumulative INR 2,998,668.80 |

Why ties cause the Q6 difference: tied customers get the same RANK() (and it then skips numbers), whereas ROW_NUMBER() always numbers rows 1, 2, 3, ... and so keeps counting through the tie.

**pandas cross-check.** `pd.pivot_table(..., margins=True)` grand total = INR 2,998,668.80, equal to the SQL total. Completed order_amount per city:

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

**EDA of completed order_amount:** mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, fences -283.50 / 645.30, outliers below 0, above 897. Correlation matrix (completed orders): order_amount-discount_pct -0.095, order_amount-quantity +0.087, order_amount-delivery_minutes -0.003, discount_pct-quantity 0.000, discount_pct-delivery_minutes -0.010, quantity-delivery_minutes -0.013.

*What is the data about?* Six months (April-September 2025) of quick-commerce orders from 1,175 buying customers across six cities. *Distributions?* Order values are right-skewed: the mean (269.91) is twice the median (135.00), skewness is 4.61. *Outliers?* 897 high orders above INR 645.30 (mostly electronics), none below; they are real big-ticket purchases, not errors. *Relationships?* The strongest pair is order_amount with discount_pct (-0.095); order_amount and quantity is +0.087. Correlations this small are weak, and even a larger one would not establish causation.

## Part B: Customer Segments and Lapse Prediction

**RFM** (snapshot 2025-10-01, completed orders only): 1,175 customers; recency median 19, min 0, max 180; frequency mean 9.46, max 33; monetary mean 2,552.06, max 14,766.55.

**K-means** on StandardScaler-scaled RFM (`n_init=10`, `random_state=42`):

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Chosen **K = 4**: inertia drops sharply up to K=4 (1268 to 743) and then flattens (743 to 617), the silhouette peaks at K=4 (0.526), and four groups are few enough for a campaign team to act on. Inertia always falls as K grows because extra centroids can only bring points closer to a centre (at K equal to the number of customers it is zero), so a lower inertia at K=8 than K=2 says nothing by itself; it has to be balanced against the elbow, the silhouette and interpretability.

**Segment profile** (cluster numbers are arbitrary labels with no order):

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | Premium High-Value (**is_premium**) | 138 | 24.1 | 6.8 | 7,129.70 | Early access, concierge support, protect with retention offers |
| 2 | Loyal Regulars | 419 | 8.8 | 19.4 | 3,410.02 | Subscription / bundle offers, loyalty points |
| 0 | Occasional Shoppers | 436 | 28.9 | 3.8 | 1,057.62 | Nudge to a second weekly basket with small coupons |
| 3 | Dormant / At-Risk | 182 | 109.0 | 2.2 | 686.0 | Win-back campaign, limited-time discount |

Names are assigned in code from the profile: Premium = highest mean monetary. **PCA** (2 components) explained variance 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium` (138 premium rows).

**Lapse problem.** Cutoff 2025-09-01; cohort = customers with a completed order before the cutoff: **1,156** customers, **443 lapsed (38.32%)** (no completed order 1-30 Sep). Split `test_size=0.2, stratify=y, random_state=42`: 924 train / 232 test. Features (orders before the cutoff only): recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure days. Baseline "always not lapsed" on test: accuracy 61.6%, lapsed-class F1 0.00.

**Leakage statements.** No order dated 2025-09-01 or later entered any feature (asserted in code); the segment labels from the RFM clustering were not used as features; every scaler, one-hot encoder and the over-sampler was fitted/used on training rows only and merely applied to test rows (everything sits inside a scikit-learn `Pipeline`).

**Preprocessing.** `StandardScaler` for numeric columns and `OneHotEncoder(drop="first")` (k-1 columns) for city, gender, favourite category. Learned mean of frequency in the scaler (train rows) = 8.115, full-cohort mean = 8.029; they differ because the scaler only sees the 924 training rows, a sample of the cohort. Missing values in any feature column: 0 (asserted).

**Tuning (GridSearchCV, cv=5, scoring f1).**

| Model | Grid | Best params | Best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C in {0.01, 0.1, 1, 10, 100} | C = 100 | 0.651 |
| RandomForest(n_estimators=100, random_state=42) | max_depth {3,5,8} x min_samples_leaf {5,10,20} | depth 8, leaf 10 | 0.681 |

**Test set (lapsed = positive class)**, confusion matrix as [[TN, FP], [FN, TP]]:

| Model | Confusion matrix | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline | all predicted not-lapsed | 0 | 0 | 0.00 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30], [31, 58]] | 0.659 | 0.652 | 0.655 |

**Selected model: random forest** (test F1 0.655, above the 0.55 requirement). It has the higher cross-validated F1 (0.681 vs 0.651) and slightly higher precision at the same recall. Missing a customer who is about to lapse costs the lifetime revenue of that customer, whereas wrongly targeting a loyal one costs only a coupon, so recall matters more than precision here, and the forest catches as many lapsing customers (58 of 89) as the logistic model with fewer false alarms (30 vs 32). The test set was used only after tuning finished. The best logistic C sits at the edge of the grid, so a wider grid could be tried.

**Class-balance experiment** (training lapsed rows duplicated with replacement, `random_state=42`, to 570 / 570; best logistic regression refit; test set unchanged):

| Logistic regression | Precision | Recall | F1 | Confusion matrix |
|---|---|---|---|---|
| Un-balanced | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| Over-sampled | 0.589 | 0.854 | 0.697 | [[90, 53], [13, 76]] |

Recall moved most (+0.20, from 58 to 76 of 89 lapsing customers caught), while precision fell (0.644 to 0.589). For the retention team this means a broader net: more lapsing customers reached, at the price of about 21 more loyal customers receiving an offer they did not need. Worth it when offers are cheap relative to customer value.

**Interpretation.** Top-5 random-forest importances: frequency 0.280, monetary 0.271, recency 0.115, average delivery minutes 0.061, cancellation rate 0.053 (buying habit dominates; city and demographics barely matter). Logistic coefficients (standardised features): frequency **-1.70** (the more often someone orders, the less likely they lapse); monetary **+0.28** (given the same frequency, big-ticket buyers, mostly infrequent electronics shoppers, lapse more); cancellation rate **-0.12** (tiny and not a reliable signal).

**Persistence.** The selected pipeline is saved to `outputs/lapse_model.joblib`; the reloaded model reproduces the test predictions and probabilities exactly (checked in code).

**Retention list.** 40 test-set customers are in the Premium segment; the 10 with the highest predicted lapse probability are in `outputs/retention_list.csv` (`customer_id, lapse_probability, actually_lapsed`); **1 of the 10 truly lapsed**. The Premium flag is a post-hoc whole-window label (its RFM includes September, which defines the lapse label), so this list is illustrative and not a leakage-free evaluation.

## Part C: Support Ticket Triage with Guardrails

Pipeline: `src/c_triage.py`; tickets are processed in file order with `keep_default_na=False`.

**Input guardrails.** 4 empty tickets get `skipped_empty`; 4 long tickets (each estimated at 734 tokens, limit `MAX_INPUT_TOKENS = 400`) get `flagged_long`; neither group is ever sent to the model.

**Redaction** (email, then phone, then order id, with the specified regexes), across all 164 tickets:

| Type | Substitutions |
|---|---|
| email -> [EMAIL] | 55 |
| phone -> [PHONE] | 91 |
| order id -> [ORDER_ID] | 132 |
| **total** | **278** |

Re-applying the three patterns to the redacted texts finds 0 matches. Every system and user message actually sent is asserted PII-free in code, and the user message is exactly the redacted ticket text.

**Prompts.** Both contain CLASSIFY, the four categories and the JSON-only format. Zero-shot (about 36 estimated tokens) has no `EXAMPLES:` line; few-shot (about 111 tokens) has an `EXAMPLES:` block with four labelled examples including a fraud one.

**Retries.** Every call goes through a retry wrapper (max 3 retries, wait = 0.05 x 2^n x uniform(0.9, 1.15) s, one log line per retry). `mock_llm.reset()` runs before every full run. A full run makes 156 model calls and meets 19 `TransientLLMError`s, all recovered by one retry (19 retry log lines, waits between 0.0453 and 0.0562 s); no ticket ended in a permanent error.

**Validation.** JSON with an allowed category = `classified_json`; non-JSON text is scanned with `re.findall(r"[a-z]+", ...)` and accepted if exactly one allowed category word appears = `classified_keyword`; everything else (including JSON with category "shipping") = `flagged_invalid_output`, sent to a human, never guessed. Status counts for one full run (identical for both prompts): **135 classified_json, 12 classified_keyword, 9 flagged_invalid_output**, plus 4 skipped_empty and 4 flagged_long.

**Zero-shot versus few-shot** (no budget):

| Run | Tickets classified | Accuracy | Total tokens | Tokens / classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126 of 147) | 10,024 | 68.2 |
| Few-shot | 147 | 100.0% (147 of 147) | 21,724 | 147.8 |

Zero-shot confusion table (rows = true, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

When the few-shot tokens are worth it: the examples add about 80 tokens per classified ticket (21,724 vs 10,024 tokens, 2.2 times the cost) but remove all 21 zero-shot errors. Eleven of those errors are fraud tickets read as billing, which would miss the HIGH-priority security route, so the extra spend is clearly justified for fraud-sensitive triage; for a low-stakes pipeline where a 14% error rate is tolerable, zero-shot is cheaper.

**Token budget.** With zero-shot and `TOKEN_BUDGET = 6000` the pipeline spent 6,044 tokens and set **62 tickets to `budget_exhausted`** (human review); saved to `outputs/triage_budget_demo.csv` (budget used: 6000).

**Final routing** (`outputs/triage_results.csv`, 164 rows, few-shot run, `TOKEN_BUDGET = 100000`, segments read from Part B; customers absent from the segment file count as not premium): human_review 17 (9 invalid outputs + 4 empty + 4 long), security_team 34 (HIGH), premium_desk 11 (HIGH), support_queue 102 (NORMAL); 45 HIGH and 119 NORMAL in total.

**What was rule-validated and what went to humans.** 147 tickets passed the JSON / keyword validation rules and were routed automatically; 17 tickets (9 invalid model outputs, 4 empty, 4 over-length) went to human review.

## Part D: Policy RAG Assistant and Refund Agent

**Chunking** (`src/d_rag.py`, 40 words, overlap 10): 15 chunks in total: refund 4, delivery 3, escalation 3, payment 3, privacy 2, each stored with its source file name.

**Retrieval.** `TfidfVectorizer()` fitted on the chunks, queries transformed with the same vectorizer, ranked by cosine similarity; `retrieve(query, k=3)` returns `source, score, text`. For "refund policy late delivery" all three top chunks come from `refund_policy.txt`. TF measures how often a word occurs within a chunk; IDF down-weights words that appear in many chunks so rare, distinctive words count more. Limitation: TF-IDF matches words, not meaning, so "money back" would not match "refund".

**Answering.** `answer(question)` retrieves 3 chunks and sends a system message containing POLICY ASSISTANT (answer only from the context, otherwise say it could not find the information) and a user message `CONTEXT:\n[file]\nchunk ... QUESTION:\n<question>` to `mock_llm.chat`, returning the text and the sources.

**Evaluation** (`policy_eval.csv`, 12 answerable + 2 unanswerable):

| Setting | Chunks | expected_doc in top 3 (of 12) | expected_fact in answer (of 12) | Fallback on unanswerable (of 2) |
|---|---|---|---|---|
| Overlap 10 | 15 | 12 | 12 | 2 |
| Overlap 0 | 13 | 12 | 10 | 2 |

With zero overlap two questions break: "What delivery time makes a completed order eligible for a full refund?" (the answer "more than 45 minutes" is split across a chunk boundary, so the model returns the sentence fragment "receives a full refund of the order amount.") and "How soon is money returned when a payment fails but is deducted?" (the "48 hours" part falls in the next chunk). Overlap keeps a sentence whole in at least one chunk.

**Tools** (`src/d_agent.py`): `TOOLS_SPEC` lists five tools (name, one-sentence description, JSON-style parameters) and `TOOL_MAP` maps names to functions: `get_order` (reads `data/instamart.db`), `search_policy` (uses `retrieve`), `calculate_refund` (full order amount if delivered in more than 45 minutes, else 0), `issue_refund` and `escalate_to_human` (each append a timestamped line with the tool name to `outputs/refund_ledger.jsonl`).

**ReAct loop.** `run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` calls the model, executes the requested tool by name from `TOOL_MAP`, appends the assistant and tool messages, and loops until a final answer. The normal status is the word before the colon. Results for the 7 scenarios (`outputs/agent_runs.jsonl`, one line each):

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,047 |
| S2 | REFUND | 135.00 | 5 | 1,047 |
| S3 | ESCALATE | 1999.00 | 5 | 1,034 |
| S4 | DENIED | 0.00 | 4 | 725 |
| S5 | DENIED | 0.00 | 3 | 427 |
| S6 | NOT_FOUND | 0.00 | 2 | 132 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,180 |

All refund amounts equal `expected_refund_inr`. S7 uses all six model calls under the default `token_limit=6000`.

**Stop safely.** Limits: max iterations (S7 above), token limit (checked after every model call; stops without running the requested tool), and wall-clock limit (`time.monotonic()`, checked at the top of each iteration). Every fallback returns the steps so far and a short human-handoff string. Demonstrations (`outputs/agent_demo_runs.jsonl`):

| Demo | Status | Detail |
|---|---|---|
| S1 with `token_limit=200` | **FALLBACK_TOKEN_LIMIT** | 3 model calls, 444 tokens, only get_order and search_policy executed |
| S1 with `time_limit_seconds=0` | **FALLBACK_TIMEOUT** | 0 model calls, 0 tools executed |

**Resumable runs.** After each tool observation a checkpoint `state/<run_id>.json` is written (run id, scenario id, full message list, tokens used, step number, tools executed, steps, limits). `resume_agent(run_id)` reloads it and continues (the wall-clock timer restarts on resume). Demo: S1 with a simulated crash right after the second tool's checkpoint, then resume. Tools executed across the crash and the resume, in order: **get_order, search_policy, calculate_refund, issue_refund** (four executions, none repeated); final status REFUND with INR 128.25 (5 model calls in total). `outputs/refund_ledger.jsonl` holds 4 lines: S1 issued 128.25, S2 issued 135.00, S3 escalated 1999.00, and the crash-demo S1 issue (run id `S1-crash-demo`).

**ReAct versus plan-and-execute.** My ReAct agent asked the model at every step: 5 calls for S1 (1,047 tokens), 3 for S5, 2 for S6. Each call re-sends the growing history, so tokens climb with each step. A plan-and-execute design asks once for the whole step list and then runs it, so it would need about 1 to 2 calls and fewer tokens for the happy path (I did not build or measure it; this is an estimate). The difference shows on unexpected observations: in S6 ReAct saw "order not found" after 2 calls and stopped, and in S5 it saw a cancelled order and denied. A fixed plan would already contain "calculate refund, issue refund" steps that make no sense for those cases, so it needs an error check and a replanning call, which removes much of the saving. Since this workflow moves money, branches on what each step returns (not found, cancelled, on time, above INR 1000) and is only about 5 steps long, I would choose ReAct with the three hard limits: its decisions are always based on the latest observation, and the extra tokens (about 1,000 per refund) are small. Plan-and-execute becomes attractive for long, predictable workflows with many steps.

## Limitations and Next Steps

- All data is synthetic and `mock_llm.py` is a rule-based stand-in; real model behaviour (formats, errors, cost) will differ, so accuracy numbers (100% few-shot) are not transferable.
- The Premium flag is a post-hoc whole-window label (its RFM includes September, which defines the lapse label), so the retention list is illustrative and not a leakage-free evaluation. A proper version would compute segments only from data before the cutoff.
- The lapse model uses one time split and one cutoff; a rolling-window validation would be more reliable. The best logistic C is at the edge of the grid; a wider grid and calibrated probabilities are next steps.
- Missing delivery times were imputed with category medians and missing ages with city medians (rounded); this is a modelling assumption.
- The refund agent assumes every request is inside the 7-day window and does not verify that the requester owns the order; real use needs both checks and a human approval step for escalations.
- TF-IDF retrieval is word-based; embeddings would handle paraphrases better.
- The package versions in `requirements.txt` were captured from the build environment; regenerate with `pip freeze` in your own virtual environment if your Python version differs.
