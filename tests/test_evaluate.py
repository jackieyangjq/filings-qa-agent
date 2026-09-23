import json

import pytest

from filings_qa import evaluate
from filings_qa.evalset import Question
from filings_qa.evaluate import JUDGE_PROMPT, JudgeSchema, collect, report, run, section_of
from filings_qa.llm import FakeLLM
from filings_qa.store import Hit

# Chunks of conftest's store.
G1 = "ACME-10-Q-20240802-2-001"  # "Data center revenue growth was strong: ..."
G2 = "ACME-10-Q-20240802-2-002"  # "Gaming revenue declined because of weaker consumer demand."
G3 = "ACME-10-Q-20240802-1A-001"  # "Supply chain disruptions could delay shipments of our products."
OTHER = "OTHR-10-K-20240216-7-001"


def f(i):
    """A filler id: item 9 of the ACME 10-Q, which is not stored (the answer model is not shown it)."""
    return f"ACME-10-Q-20240802-9-{i:03d}"


def fill(*head, n=10):
    return [*head, *(f(i) for i in range(1, n - len(head) + 1))]


QUESTIONS = [
    Question("q1", "How did ACME's data center revenue do in Q2 2024?", "It grew strongly.",
             "Data center revenue growth was strong", G1, [G1], "ACME-10-Q-20240802", "2", "ACME", "ACME/2"),
    Question("q2", "Why did ACME's gaming revenue decline in Q2 2024?", "Weaker consumer demand.",
             "Gaming revenue declined because of weaker consumer demand.", G2, [G2], "ACME-10-Q-20240802", "2",
             "ACME", "ACME/2"),
    Question("q3", "What could delay ACME's shipments?", "Supply chain disruptions.",
             "Supply chain disruptions could delay shipments of our products.", G3, [G3], "ACME-10-Q-20240802", "1A",
             "ACME", "ACME/1A"),
    Question("q4", "How many subscribers did Netflix report for Q2 2024?", "(none: Netflix is not in the corpus)",
             ticker="NFLX", kind="other_company", source_qid="q1", answerable=False),
]

# The ranking of each question by each strategy. Gold ranks: bm25 1, 7, missing; dense 3, 2, 6.
RANKINGS = {
    "bm25": {
        "q1": fill(G1),
        "q2": [f(1), G1, f(2), f(3), f(4), OTHER, G2, f(5), f(6), f(7)],  # G1 is in the gold section, at 2
        "q3": fill(G1, G2),
        "q4": fill(OTHER, G1),
    },
    "dense": {
        "q1": fill(f(11), f(12), G1),
        "q2": fill(f(11), G2),
        "q3": fill(f(11), f(12), f(13), f(14), f(15), G3),
        "q4": [G1, OTHER, f(1), f(2), f(3), f(4), f(5), f(6), G3, f(7)],  # G3 is 9th: not shown to the model
    },
}


class FakeRetriever:
    """Ranks each question's chunks as ``RANKINGS`` says; records its calls."""

    def __init__(self, rankings=RANKINGS):
        self.rankings = rankings
        self.qid = {q.question: q.qid for q in QUESTIONS}
        self.calls = []

    def __call__(self, query, strategy, k):
        self.calls.append((self.qid[query], strategy, k))
        return [Hit(c, 1.0 / rank, rank) for rank, c in enumerate(self.rankings[strategy][self.qid[query]][:k], 1)]


def reply(*sentences, abstained=False):
    return {"abstained": abstained, "sentences": [{"text": t, "citations": c} for t, c in sentences]}


def grade(verdict):
    return {"reason": f"The answer is {verdict}.", "verdict": verdict}


ANSWERS = [
    # bm25
    reply(("Data center revenue growth was strong.", [G1])),
    reply(
        ("Gaming revenue declined on weaker demand.", [G2]),
        ("Data center revenue grew as well.", [OTHER]),  # another filing: not OK
        ("Data center revenue growth was strong.", [G1]),  # not the gold chunk, but its section: OK
    ),
    reply(("The excerpts do not say what could delay shipments.", []), abstained=True),
    reply(("The excerpts do not cover Netflix.", []), abstained=True),
    # dense
    reply(("Data center revenue growth was strong.", [G1])),
    reply(("Weaker consumer demand.", [G2])),
    reply(("Shipments could slip for lack of demand.", [G3])),
    reply(("Netflix reported 300 million subscribers.", [G1])),  # should have abstained
]
GRADES = [grade("correct"), grade("partial"), grade("correct"), grade("correct"), grade("incorrect")]


def run_all(store, cache_dir, answers, grades, strategies=("bm25", "dense"), questions=QUESTIONS, **kwargs):
    llm = FakeLLM(answers, tokens=(1000, 200), latency=0.5)
    judge_llm = FakeLLM(grades, tokens=(300, 20))
    results = run(questions, list(strategies), retriever=FakeRetriever(), llm=llm, judge_llm=judge_llm, store=store,
                  cache_dir=cache_dir, **kwargs)
    return results, llm, judge_llm


def test_metrics_of_two_strategies_match_a_hand_count(store, tmp_path):
    results, llm, judge_llm = run_all(store, tmp_path / "cache", ANSWERS, GRADES)
    assert results.complete and not results.stopped
    bm25, dense = results.summary["bm25"], results.summary["dense"]

    # recall: bm25 finds q1 at 1 and q2 at 7, misses q3; dense finds q1 at 3, q2 at 2, q3 at 6
    assert (bm25["recall_at_5"], bm25["recall_at_10"]) == (0.3333, 0.6667)
    assert (dense["recall_at_5"], dense["recall_at_10"]) == (0.6667, 1.0)
    # section: bm25 q2 has G1 (same filing, item 2) at rank 2; dense q3's only 1A chunk is at 6
    assert (bm25["section_hit_at_5"], dense["section_hit_at_5"]) == (0.6667, 0.6667)
    # grades over the 3 answerable questions: bm25 correct, partial, abstained; dense correct, correct, incorrect
    assert (bm25["correct"], bm25["partial"], bm25["incorrect"]) == (0.3333, 0.3333, 0.3333)
    assert (dense["correct"], dense["partial"], dense["incorrect"]) == (0.6667, 0.0, 0.3333)
    assert (bm25["false_abstain"], dense["false_abstain"]) == (0.3333, 0.0)
    # the Netflix question: bm25 abstained, dense answered
    assert (bm25["abstain_ok"], dense["abstain_ok"]) == (1.0, 0.0)
    # bm25: q1's sentence cites G1; q2's cite G2 (ok), OTHR (another filing) and G1 (the gold section): 3 of 4
    assert (bm25["citation_ok"], bm25["cited_sentences"]) == (0.75, 4)
    assert (dense["citation_ok"], dense["cited_sentences"]) == (1.0, 3)
    # tokens and cost of the 4 answers: 4 x (1000 x $0.30 + 200 x $2.50) / 1M
    assert (bm25["avg_in_tokens"], bm25["avg_out_tokens"], bm25["est_cost_usd"]) == (1000.0, 200.0, 0.0032)
    assert 0.5 <= bm25["avg_latency_s"] < 0.6
    assert (bm25["judge_calls"], bm25["judge_in_tokens"], dense["judge_calls"], dense["judge_out_tokens"]) == (2, 600,
                                                                                                             3, 60)
    assert (bm25["done"], bm25["answerable"], bm25["unanswerable"], bm25["failed"]) == (4, 3, 1, 0)

    # the judge saw the question, the reference answer and quote, and the answer; nothing was graded twice
    assert len(judge_llm.calls) == 5 and judge_llm.script == [] and llm.script == []
    first = judge_llm.calls[0]
    assert first["config"]["response_schema"] is JudgeSchema and first["config"]["system_instruction"] == JUDGE_PROMPT
    assert first["contents"] == (
        f"Question: {QUESTIONS[0].question}\n\nReference answer: It grew strongly.\n\nPassage of the filing: Data"
        " center revenue growth was strong\n\nSystem's answer: Data center revenue growth was strong."
    )
    assert llm.calls[0]["models"] == judge_llm.calls[0]["models"] == ["gemini-3.5-flash-lite"]
    # the answer model saw the first 8 of the 10 ranked chunks (the fillers are not stored)
    assert G2 in llm.prompts[1] and OTHER in llm.prompts[1] and f(1) not in llm.prompts[1]
    assert G1 in llm.prompts[7] and G3 not in llm.prompts[7]  # dense q4: G3 was ranked 9th

    records = {(r.strategy, r.qid): r for r in results.records}
    assert (records["bm25", "q2"].rank, records["bm25", "q2"].section_rank) == (7, 2)
    assert records["bm25", "q3"].verdict == "incorrect" and records["bm25", "q3"].reason == "abstained"
    assert records["dense", "q4"].verdict is None and records["dense", "q4"].abstained is False


def test_a_second_run_reads_the_cache_and_asks_no_model(store, tmp_path):
    cache = tmp_path / "cache"
    first, _, _ = run_all(store, cache, ANSWERS, GRADES)
    saved = json.loads((cache / "bm25" / "q2.json").read_text())
    assert saved["ranking"] == RANKINGS["bm25"]["q2"] and saved["judge"]["verdict"] == "partial"
    assert (saved["k"], saved["depth"]) == (8, 10)

    again, llm, judge_llm = run_all(store, cache, [], [])  # any model call would fail
    assert llm.calls == [] and judge_llm.calls == []
    assert again.summary == first.summary and all(r.cached for r in again.records)
    assert collect(QUESTIONS, ["bm25", "dense"], cache_dir=cache).summary == first.summary  # --report-only


def test_a_changed_question_is_not_read_from_the_cache(store, tmp_path):
    cache = tmp_path / "cache"
    run_all(store, cache, ANSWERS[:4], GRADES[:2], strategies=["bm25"])
    changed = [QUESTIONS[0], Question(**{**QUESTIONS[1].__dict__, "question": QUESTIONS[1].question + "?"})]
    retriever = FakeRetriever({"bm25": {"q1": RANKINGS["bm25"]["q1"], "q2": RANKINGS["bm25"]["q2"]}})
    retriever.qid[changed[1].question] = "q2"
    llm, judge_llm = FakeLLM([ANSWERS[1]]), FakeLLM([grade("correct")])
    results = run(changed, ["bm25"], retriever=retriever, llm=llm, judge_llm=judge_llm, store=store, cache_dir=cache)
    assert [r.cached for r in results.records] == [True, False] and len(llm.calls) == 1


class Err(Exception):
    def __init__(self, code, text=""):
        super().__init__(text or f"code {code}")
        self.code = code


PER_MINUTE = Err(429, "429 RESOURCE_EXHAUSTED. {'error': {'details': [{'violations': [{'quotaId': "
                      "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}]}, {'retryDelay': '7s'}]}}")
PER_DAY = Err(429, "429 RESOURCE_EXHAUSTED. {'error': {'details': [{'violations': [{'quotaId': "
                   "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}]}}")


def test_a_failed_item_is_not_saved_and_is_tried_again(store, tmp_path):
    cache = tmp_path / "cache"
    answers = [ANSWERS[0], Err(400, "bad request"), ANSWERS[2], ANSWERS[3]]
    results, llm, _ = run_all(store, cache, answers, GRADES[:1], strategies=["bm25"])
    assert [r.status for r in results.records] == ["done", "failed", "done", "done"]
    assert results.records[1].error.startswith("request rejected (HTTP 400)") and not results.complete
    assert not (cache / "bm25" / "q2.json").exists()
    assert results.summary["bm25"]["done"] == 3 and results.summary["bm25"]["failed"] == 1

    again, llm, judge_llm = run_all(store, cache, [ANSWERS[1]], [GRADES[1]], strategies=["bm25"])
    assert again.complete and len(llm.calls) == 1 and "gaming" in llm.prompts[0].lower()
    assert [r.cached for r in again.records] == [True, False, True, True]


def test_a_failed_grade_keeps_the_answer_and_only_the_grade_is_asked_again(store, tmp_path):
    cache = tmp_path / "cache"
    results, _, _ = run_all(store, cache, ANSWERS[:4], [grade("correct"), Err(503)], strategies=["bm25"],
                            sleep=lambda s: None)
    assert [r.status for r in results.records] == ["done", "failed", "done", "done"]
    assert json.loads((cache / "bm25" / "q2.json").read_text())["judge"] is None  # the answer is kept

    again, llm, judge_llm = run_all(store, cache, [], [grade("partial")], strategies=["bm25"])
    assert again.complete and llm.calls == [] and len(judge_llm.calls) == 1
    assert again.summary["bm25"]["partial"] == 0.3333


def test_a_per_minute_limit_is_waited_out(store, tmp_path):
    waits = []
    answers = [PER_MINUTE, ANSWERS[0], *ANSWERS[1:4]]
    results, llm, _ = run_all(store, tmp_path, answers, GRADES[:2], strategies=["bm25"], sleep=waits.append)
    assert results.complete and waits == [8.0]  # the 7 s Gemini asked for, plus one


def test_a_used_up_daily_quota_stops_the_run_and_a_rerun_goes_on(store, tmp_path):
    cache = tmp_path / "cache"
    results, llm, _ = run_all(store, cache, [ANSWERS[0], PER_DAY], GRADES[:1], sleep=lambda s: None)
    assert [r.status for r in results.records] == ["done", "failed"] + ["not run"] * 6
    assert "daily quota" in results.stopped and results.summary["dense"]["not_run"] == 4
    assert len(llm.calls) == 2  # nothing asked after the daily quota ran out

    again, llm, _ = run_all(store, cache, ANSWERS[1:], GRADES[1:])
    assert again.complete and len(llm.calls) == 7
    assert again.summary["bm25"]["correct"] == 0.3333


def test_the_run_stops_after_three_failures_in_a_row(store, tmp_path):
    results, llm, _ = run_all(store, tmp_path, [Err(500)] * 9, [], sleep=lambda s: None)
    assert [r.status for r in results.records] == ["failed"] * 3 + ["not run"] * 5
    assert "3 failures in a row" in results.stopped

    # three failures that are not in a row do not stop it
    answers = [Err(500), ANSWERS[1], Err(500), ANSWERS[3], Err(500), *ANSWERS[5:]]
    results, llm, _ = run_all(store, tmp_path / "2", answers, [GRADES[1], GRADES[3], GRADES[4]])
    assert [r.status for r in results.records] == ["failed", "done", "failed", "done", "failed", "done", "done", "done"]
    assert not results.stopped


def test_report_has_a_row_per_strategy(store, tmp_path):
    results, _, _ = run_all(store, tmp_path, ANSWERS, GRADES)
    lines = report(results).splitlines()
    assert lines[0].startswith("| Strategy | Done | Recall@5 | Recall@10 | Section hit@5 | Correct |")
    assert lines[2].startswith(
        "| bm25 | 4/4 | 33.3% | 66.7% | 66.7% | 33.3% | 33.3% | 33.3% | 75.0% | 100.0% | 33.3% | 1,000 | 200 | 0.5"
    )
    assert lines[2].endswith(" | $0.0032 |")
    assert lines[3].startswith("| dense | 4/4 | 66.7% | 100.0% | 66.7% | 66.7% | 0.0% | 33.3% | 100.0% | 0.0% | 0.0%")
    text = "\n".join(lines)
    assert "3 answerable questions" in text and "1 unanswerable" in text and "free tier" in text
    data = json.loads(json.dumps(results.to_dict()))
    assert data["settings"]["answer_models"] == ["gemini-3.5-flash-lite"] and len(data["items"]) == 8


def test_report_without_prices_shows_the_free_tier(store, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "PRICING", None)
    results, _, _ = run_all(store, tmp_path, ANSWERS[:4], GRADES[:2], strategies=["bm25"])
    assert report(results).splitlines()[2].endswith(" | free tier ($0) |")


def test_any_chunk_holding_the_quote_counts_as_retrieved():
    """A quote can sit in several chunks (overlaps, a 10-Q repeating the 10-K): the first of them in the ranking
    gives the rank, and citing any of them is a correct citation."""
    q = Question("q1", "How did ACME's data center revenue do?", "It grew.", "Data center revenue growth was strong",
                 G1, [G1, OTHER], "ACME-10-Q-20240802", "2", "ACME")
    answer = reply(("It grew strongly.", [OTHER]))
    answer.update(usage={"in": 10, "out": 2}, latency_s=0.1, model="m", dropped_sentences=0)
    grade_ = {"verdict": "correct", "reason": "ok", "model": "m", "usage": {"in": 5, "out": 1}}
    entry = {"question": q.question, "k": 8, "depth": 10, "ranking": [f(1), OTHER, G1], "retrieval_s": 0.0,
             "answer": answer, "judge": grade_}
    record = evaluate.score(q, "bm25", entry)
    assert (record.rank, record.section_rank) == (2, 2)
    assert (record.cited_sentences, record.citation_ok_sentences) == (1, 1)


@pytest.mark.parametrize(
    "chunk_id, section",
    [
        ("AAPL-10-Q-20260501-II-2-001", ("AAPL-10-Q-20260501", "II-2")),
        ("AAPL-10-K-20251031-1A-013", ("AAPL-10-K-20251031", "1A")),
        ("JPM-10-Q-20260806-0-120", ("JPM-10-Q-20260806", "")),
    ],
)
def test_section_of_reads_filing_and_item_from_a_chunk_id(chunk_id, section):
    assert section_of(chunk_id) == section
