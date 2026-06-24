# Natural Language-to-SQL Agentic Pipeline

A from-scratch end-to-end agentic NL→SQL pipeline built with `LangChain`, `Claude`, and `Voyage AI` — evaluated on the BIRD Mini-Dev benchmark. The project is structured as follows:
* **Stage A**: Core agent on Chinook — clean and small, good for learning the fundamentals without BIRD's real-world messiness getting in the way
* **Stage B**: Self-correction loop (still on Chinook)
* **Stage C**: Eval harness, using BIRD — pick a handful of its (question, gold SQL) pairs, measure accuracy
* **Stage D**: Schema retrieval — pick one of BIRD's larger/messier databases, show accuracy with vs. without retrieval
* **Stage E**: Gradio app → deployed to [Hugging Face Spaces](https://huggingface.co/spaces/anantshri1/nl2sql-agentic-pipeline)

---
## Manual NL→SQL chain

### Core Agent on [Chinook Database](https://github.com/lerocha/chinook-database)

Dependencies installed: `langchain, langchain-anthropic, langchain-community, sqlalchemy`.

The original plan was to download a prebuilt `Chinook.sqlite` file from a Google Cloud Storage bucket referenced in LangChain's own tutorial, but that asset no longer exists (confirmed via a `NoSuchKey` XML error after inspecting the tiny downloaded file). The fix was to build the database differently and more durably: fetch the actual SQL script (`CREATE TABLE` and `INSERT` statements as plain text) from the actively maintained `lerocha/chinook-database` GitHub repository, execute it into an in-memory `SQLite` connection via Python's `sqlite3` module, and wrap that connection in a `SQLAlchemy` engine using `StaticPool` with a creator lambda. This was necessary because an in-memory SQLite database is tied to a single connection object — without `StaticPool` forcing reuse of that exact connection, the data would vanish after the first query. The resulting engine was wrapped in LangChain's `SQLDatabase` utility, which exposes the schema (`table names, columns, sample rows`) as a text string via `get_table_info` — this is how the LLM later learns what's in the database.

The first real NL-to-SQL chain was then built using `LangChain` Expression Language: 
* a `ChatPromptTemplate` piped into the LLM (`prompt | llm`), with the database schema and dialect injected into the prompt and the natural-language question as the human message;
* a `clean_sql` helper function was added to strip markdown code fences, since LLMs sometimes wrap SQL in triple-backtick fences despite explicit instructions not to;
* an `ask_db` function combined these: build the schema, invoke the chain, clean the output, execute it via `db.run`, and return the result. 

> Tested successfully on "How many tracks are there in total?", returning a correct count.

---
### Self-Correction Loop

The correction logic was built and tested in isolation before being wired into a full loop, specifically because Claude at `temperature 0` reliably succeeds on Chinook's small schema on the first attempt — meaning a full retry loop run on normal questions might never actually exercise the correction path. To force a test of the correction path, a deliberately broken query (`SELECT COUNT(*) FROM Tracks`, using the wrong plural table name) plus its real `SQLite error` message were manually fed into a correction chain.

With the correction logic validated, it was wired into `ask_db_with_retry`: generate an initial query, then loop up to `max_retries` times, attempting execution via `db.run` inside a try block; on any exception, the error message is fed into the correction chain to produce a new query for the next attempt. A successful run returns immediately from inside the loop; exhausting all retries raises a RuntimeError rather than silently returning a wrong answer. Tested successfully on Chinook, succeeding on the first attempt with no correction triggered (expected, given Chinook's simplicity).

---

## BIRD Mini Dev Benchmarking

The existing `ask_db_with_retry` pipeline used `db.run()`, which returns query results as a stringified Python list (e.g. "`[(0.0657...,)]`") rather than structured data. This is fine for feeding results back to an LLM, but useless for programmatic grading, since comparing stringified floats produces false negatives from formatting alone. The fix was a dedicated grading helper, `execute_for_grading(db, sql)`, which calls `SQLDatabase`'s `internal _execute()` method instead of `run()` to get real Python tuples with correct types preserved. This function is used only for grading and never touches the existing agent-facing retry loop, keeping that path unchanged.

On top of that, `is_correct(predicted_sql, gold_sql, db)` was built to do execution-accuracy comparison. A deliberate methodological choice was made here: exact ordered-tuple comparison was used instead of BIRD's standard set-based comparison (which ignores row order and duplicates). This is stricter than the official BIRD metric, meaning the accuracy numbers in this project are a conservative lower bound relative to what BIRD's own leaderboard would report for equivalent SQL. The function treats any predicted SQL that fails to execute as simply incorrect, rather than raising, since a single bad query should never crash a 30-example evaluation run.

`ask_db_with_retry` was also adjusted at this point: previously it returned the executed result string, but the eval loop needs the actual SQL text to regrade it independently, so it was changed to return the SQL string on success while still validating execution internally (to keep the self-correction retry trigger working), and to keep raising `RuntimeError` on total failure after `max_retries`.

With those pieces in place, `run_eval(examples, db, max_retries)` was built as the main evaluation loop: it iterates over a list of BIRD examples, generates a prediction for each via `ask_db_with_retry`, grades it with `is_correct`, and stores a full per-example result dictionary (`question_id, question, predicted_sql, gold_sql, correct`) rather than just a running tally, specifically so failures can be inspected afterward rather than only knowing a final percentage. RuntimeError from total generation failure is caught at this loop level (not inside `ask_db_with_retry` itself), keeping the division of responsibility clean: the generation function stays loud about failure, the eval loop is what knows one bad example shouldn't kill the other 29.

The first full run, on the `debit_card_specializing` database (30 examples, 5 tables: customers, gasstations, products, transactions_1k, yearmonth), produced **43.3% accuracy (13/30 correct)**. Notably, every single example executed successfully on attempt 1 — meaning the self-correction retry loop, while functioning correctly, never actually fired during this run, because all 17 failures were syntactically valid SQL that computed the wrong answer. This is an important finding in itself: retry-on-error can only catch one class of failure (queries that don't execute) and is structurally blind to semantically-wrong-but-valid SQL, which turned out to be the dominant failure mode.
A substantial amount of this conversation was spent manually inspecting failures to build an empirically-grounded failure taxonomy, rather than guessing at categories. A show_failure helper was built to print a failing example's question, evidence, predicted SQL, gold SQL, and both actual result sets side by side. Going through roughly 17-18 failures (the exact count varied slightly between repeated runs due to model non-determinism, discussed below) produced seven distinct, evidence-backed failure categories:

1. Wrong aggregate / didn't follow evidence's formula literally — 1473 (`SUM` vs `AVG`)
2. Sign/semantic logic correct, but tiny floating-point rounding diff — 1476: `402524570.16999084` vs `402524570.1700022` — these differ in the 9th significant digit. This is a new category: floating-point accumulation order differences (`CASE/SUM` vs `IIF/SUM` can sum in different order → tiny float drift). Under strict ordered-tuple equality, this scores as wrong despite being correct to 8+ significant figures.
3. Format/shape mismatch, same underlying answer — 1490 ('`201304`' vs '`04`')
4. Wrong subquery/filter logic entirely, predicted answer structurally diverges — 1498 (gold picks "the min consumption row" customer, not "min per segment, then average" — predicted built a completely different multi-CTE structure than what evidence's wording wanted), 1501 (massive divergence — wrong formula scaffolding), 1505 (wrong WHERE column entirely — predicted used `Consumption > 46.73` against per-row consumption but didn't get denominator right per evidence), 1500 (interpretation difference, established already)
5. Wrong table/column joined — predicted query silently returns empty result — 1514, 1521 (need re-check from earlier list, but 1514 here: predicted filters `Date LIKE '2013-06%'` on `transactions_1k.Date`, but that table's date format apparently isn't `'2013-06%'`-shaped the way `yearmonth.Date` is, or the wrong table is being filtered—gold filters via `yearmonth` join instead, predicted's filter never matches anything, returns `[]`). Same root cause in 1505 (1505 in this batch — September 2013 product list — predicted's `strftime('%Y-%m', t.Date)` also returns `[]`, meaning `transactions_1k.Date` isn't in the format `strftime` expects, or it's the wrong table to filter by date at all — gold filters on `yearmonth.Date` matching customer, not on `transactions_1k.Date` directly).
6. Wrong domain concept — "currency" vs "nationality/country" — 1525: question asks "currency," predicted correctly returns Currency. But 1526 — question asks "nationality," predicted returned Currency ('CZK') instead of Country ('CZE'). That's a real semantic bug: nationality ≠ currency, and the model conflated them.
7. Picked the wrong specific row due to ambiguous referent ("the customer who paid X") — multiple customers/transactions matched same price — 1529: gold's own subquery resolves to a different price (1513.12) than the question states (634.8) — this looks like a likely error in BIRD's own gold SQL/dataset, not a fixable model issue. Gold result is `None`. This is worth flagging as a dataset quirk, not a model failure.
8. Column order swapped vs. gold, otherwise correct values — 1528: predicted returns (`3437.01, 537255.52`), but gold returns (`68740.20, 3437.01`) — predicted computed total spend from `transactions_1k.Price` only for that customer, while gold's total seems much larger — worth flagging as unconfirmed, needs a closer look rather than guessing.
Right concept, wrong "top" entity due to wrong ranking table — 1531 (already covered — ranked by `transactions_1k.Price` instead of `yearmonth.Consumption`)
9. Right values, wrong/extra grouping — gold collapses duplicates predicted doesn't, or vice versa — 1533: predicted returns 9 distinct customer rows; gold returns 10 rows including a duplicate 126157.7 appearing twice. Gold isn't deduplicating per customer-transaction-row; predicted used `DISTINCT` on customer. Same root issue as the unit-of-aggregation ambiguity in 1498 — "give their consumption status" read as "per person" by predicted vs "per qualifying transaction" by gold.

Summarizing these:

| No.|  Category | Detector heuristic|
| ----| ---------|-------------------|
|1| Wrong aggregate function| predicted/gold both succeed, evidence contains an explicit formula string the SQL doesn't match (hard to detect automatically — flag for manual review)|
|2| Floating-point near-miss| both numeric, both same row/col count, abs(p-g)/abs(g) < 1e-6 for all numeric cells → near-miss, not true failure |
|3| Format/shape mismatch, same magnitude| string result differs but one is a substring/reformatting of the other|
|4| Predicted returns empty result `[]` | trivially detectable — any predicted result `== []` while gold isn't|
|5| Wrong table/column for the underlying concept| hard to auto-detect; flag when result entity-IDs differ entirely (e.g. different `CustomerID` in result)|
|6|Row-count mismatch with same row content (just duplicated/deduped differently)|same `set(predicted) == set(gold)` but `sorted(predicted) != sorted(gold)` due to length, or `len` differs by small amount|
|7|Likely gold-SQL dataset error|manual flag only — can't auto-detect|

Of these seven categories, two were determined to be cheaply and reliably automatable from execution output alone: floating-point near-miss (via relative tolerance comparison, using `max(abs(gold), 1e-9)` as a denominator floor to avoid division by zero, scaled relatively rather than with a flat epsilon so it works correctly across results ranging from single digits to hundreds of millions) and empty-predicted-result (a trivial equality check against an empty list). A `categorize_failure(result, db)` function was built covering exactly these two cases, deliberately leaving everything else tagged `needs_manual_review` rather than attempting to auto-detect semantically subtle categories like wrong-table-for-concept or aggregation-unit ambiguity, since distinguishing those reliably requires understanding query meaning, not just shape. This function was wired into `run_eval` so that every failing example is automatically tagged inline during evaluation, with the tag only added when an example is actually incorrect (correct examples carry no `failure_category` key at all).

A significant secondary finding came from re-running the evaluation multiple times: despite running at `temperature=0`, results were not perfectly deterministic between runs. This was initially concerning, but was resolved by running the full evaluation three times via a new `run_eval_multiple(examples, db, n_runs, max_retries)` function, which tracks per-question_id correctness across all runs and reports both the per-run accuracy and a count of how many examples "flip" between correct and incorrect across runs. 
**The result**: accuracy was stable at exactly 43.3% across all three runs (mean 43.3%, range 43.3% to 43.3%), and only 2 of 30 examples (roughly 7%) flipped correctness at all between runs, with the other 28 fully deterministic in outcome even though `temperature` was 0. The two flipping examples were both traceable to *genuine decision-boundary sensitivity*: one flipped due to the model occasionally adding an extra descriptive column to its `SELECT` (correct core answer, but failing strict tuple comparison), and the other flipped due to the model alternating between `COUNT(*)` and `COUNT(DISTINCT)` on a question with genuine aggregation-unit ambiguity. The conclusion drawn was that `temperature=0` controls token-selection determinism but not underlying floating-point/serving-level determinism in LLM inference at scale, and that this is a known, provider-agnostic property of how LLM APIs are served, not a project-specific bug. This finding is positioned as a deliberate, quantified result rather than noise to be hidden: **accuracy is reported as stable at 43.3% plus or minus negligible run-to-run variance, with 93% of examples behaviorally deterministic**.

---
## Schema Retrieval 

---
## References
* **Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs**, Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Rongyu Cao, Ruiying Geng, Nan Huo, Xuanhe Zhou, Chenhao Ma, Guoliang Li, Kevin C.C. Chang, Fei Huang, Reynold Cheng, Yongbin Li. (2023). arXiv: [2305.03111](https://arxiv.org/abs/2305.03111)
* **A Survey of Text-to-SQL in the Era of LLMs: Where are we, and where are we going?**, Xinyu Liu, Shuyu Shen, Boyan Li, Peixian Ma, Runzhi Jiang, Yuxin Zhang, Ju Fan, Guoliang Li, Nan Tang, Yuyu Luo. (2024). arXiv: [2408.05109v6](https://arxiv.org/abs/2408.05109v6)
* **From Natural Language to SQL: Review of LLM-based Text-to-SQL Systems**, Ali Mohammadjafari, Anthony S. Maida, Raju Gottumukkala. (2024). arXiv: [2410.01066v1](https://arxiv.org/abs/2410.01066v1)
