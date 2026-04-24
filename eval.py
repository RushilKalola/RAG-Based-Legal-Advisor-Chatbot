"""
RAG Evaluation Script — Chunk-level F1 + RAGAS (Chat & Act Comparison)
=======================================================================
Place in: RAG-Based-Legal-Advisor-Chatbot/eval.py

Run:
    python eval.py                      # chunk-level only (fast)
    python eval.py --mode ragas         # RAGAS for chat
    python eval.py --mode ragas_compare # RAGAS for act comparison
    python eval.py --mode full          # everything

Install deps:
    pip install ragas==0.2.15 datasets langchain-mistralai==0.2.10 langchain-huggingface

Results are saved to: eval_results/YYYY-MM-DD_HH-MM-SS.json
so you can track improvement over time.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

# ✅ updated import
from app.services.retrieval import RetrievalService


# =============================================================================
# 1. EVAL DATASETS
# =============================================================================

@dataclass
class EvalSample:
    query: str
    relevant_kw: list[str]
    ground_truth: str
    fn_count: int = 0
    act: str = ""


@dataclass
class CompareEvalSample:
    topic: str
    act_a: str
    act_b: str
    relevant_kw_a: list[str]
    relevant_kw_b: list[str]
    ground_truth: str
    fn_count_a: int = 0
    fn_count_b: int = 0


CHAT_EVAL_DATASET: list[EvalSample] = [
    EvalSample(
    query="What is the punishment for murder under Bharatiya Nyaya Sanhita?",
    relevant_kw=["murder", "death", "imprisonment for life", "culpable homicide", "fine", "community service"],
    ground_truth=(
        "Under the Bharatiya Nyaya Sanhita, the punishments to which offenders are liable include "
        "death, imprisonment for life, rigorous imprisonment with hard labour, simple imprisonment, "
        "forfeiture of property, fine, and community service."
    ),
    fn_count=1,
    act="Bharatiya Nyaya Sanhita.pdf",
),
EvalSample(
    query="How to file a consumer complaint under Consumer Protection Act 2019?",
    relevant_kw=["complaint", "consumer", "redressal", "District Collector", "Central Authority", "investigation", "Director General"],
    ground_truth=(
        "Under the Consumer Protection Act 2019, the District Collector may, on a complaint or on a "
        "reference made by the Central Authority, take necessary action. The Director General submits "
        "inquiries and investigations to the Central Authority in the prescribed form and manner."
    ),
    fn_count=1,
    act="Consumer Protection Act 2019.pdf",
),
EvalSample(
    query="Motor vehicle accident compensation claim procedure",
    relevant_kw=["accident", "compensation", "motor vehicle", "registering authority", "police station", "insurer", "claim", "section 161"],
    ground_truth=(
        "Under the Motor Vehicles Act, a registering authority or the officer in charge of a police "
        "station shall, if required by a person claiming compensation or by an insurer, furnish "
        "particulars of the vehicle involved in the accident on payment of the prescribed fee. "
        "If an application for compensation is pending under section 161, the particulars of "
        "compensation awarded shall be forwarded to the insurer."
    ),
    fn_count=1,
    act="The Motor Vehicles Act.pdf",
),
EvalSample(
    query="What is the legal age to drive a car in India?",
    relevant_kw=["driving licence", "age", "18 years", "16 years", "motor vehicle", "transport vehicle", "50cc"],
    ground_truth=(
        "Under the Motor Vehicles Act, no person under the age of eighteen years shall drive a motor "
        "vehicle in any public place. A motorcycle with engine capacity not exceeding 50cc may be "
        "driven by a person after attaining the age of sixteen years. No person under the age of "
        "twenty years shall drive a transport vehicle in any public place."
    ),
    fn_count=1,
    act="The Motor Vehicles Act.pdf",
),
EvalSample(
    query="What is the procedure to file a civil suit in India?",
    relevant_kw=["plaint", "civil suit", "jurisdiction", "summons", "appearance", "ten days", "decree", "Order XXXVII"],
    ground_truth=(
        "Under the Code of Civil Procedure 1908, a civil suit is instituted by presenting a plaint "
        "to the court of competent jurisdiction. Once a suit is instituted, the court issues summons "
        "to the defendant to enter appearance within ten days of service. If the defendant fails to "
        "appear within ten days, the plaintiff is entitled to obtain a decree for the sum claimed."
    ),
    fn_count=1,
    act="Code of Civil Procedure.pdf",
),
]

COMPARE_EVAL_DATASET: list[CompareEvalSample] = [
    CompareEvalSample(
        topic="punishment for theft",
        act_a="Bharatiya Nyaya Sanhita",
        act_b="Information Technology Act",
        relevant_kw_a=["theft", "dishonestly", "movable property", "imprisonment", "fine"],
        relevant_kw_b=["theft", "data", "unauthorised access", "dishonestly", "imprisonment", "fine"],
        ground_truth=(
            "The Bharatiya Nyaya Sanhita defines theft as dishonest taking of movable property "
            "and prescribes imprisonment and fine for the offence. The Information Technology Act "
            "covers cyber theft such as unauthorised access to data and computer resources, "
            "prescribing imprisonment and fine under its penal provisions."
        ),
        fn_count_a=1,
        fn_count_b=1,
    ),
    CompareEvalSample(
        topic="bail provisions",
        act_a="Code of Criminal Procedure",
        act_b="Bharatiya Nyaya Sanhita",
        relevant_kw_a=["bail", "bailable", "non-bailable", "court", "surety"],
        relevant_kw_b=["bail", "bailable", "offence", "court", "release"],
        ground_truth=(
            "The Code of Criminal Procedure classifies offences as bailable and non-bailable. "
            "For bailable offences bail is a right of the accused; for non-bailable offences "
            "it is at the discretion of the court. The Bharatiya Nyaya Sanhita aligns with "
            "these principles while updating certain terminology and procedures."
        ),
        fn_count_a=1,
        fn_count_b=1,
    ),
    CompareEvalSample(
        topic="compensation for motor accident",
        act_a="Motor Vehicles Act",
        act_b="Consumer Protection Act",
        relevant_kw_a=["compensation", "tribunal", "accident", "insurance", "claim", "section 161"],
        relevant_kw_b=["compensation", "consumer", "complaint", "redressal", "commission", "damages"],
        ground_truth=(
            "The Motor Vehicles Act provides a specialised Motor Accidents Claims Tribunal route "
            "for accident compensation, where the registering authority or police station must "
            "furnish vehicle particulars to the claimant or insurer on request. "
            "The Consumer Protection Act provides redressal through District Consumer Disputes "
            "Redressal Commissions for consumer grievances including those arising from deficient "
            "services, with the District Collector empowered to act on complaints."
        ),
        fn_count_a=1,
        fn_count_b=1,
    ),
]


# =============================================================================
# 2. HELPERS
# =============================================================================

def is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def color(text: str, code: str) -> str:
    codes = {
        "green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m",
        "bold": "\033[1m", "reset": "\033[0m", "cyan": "\033[96m",
        "magenta": "\033[95m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"


def grade(val: float) -> str:
    if val >= 0.80:
        return color(f"{val:.3f}", "green")
    if val >= 0.60:
        return color(f"{val:.3f}", "yellow")
    return color(f"{val:.3f}", "red")


def save_results(results: dict, mode: str):
    os.makedirs("eval_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = f"eval_results/{timestamp}_{mode}.json"
    results["timestamp"] = timestamp
    results["mode"] = mode
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(color(f"\n  Results saved → {filepath}", "cyan"))


# =============================================================================
# 3. CHUNK-LEVEL — CHAT
# =============================================================================

# ✅ updated type hint: LegalSearchTool → RetrievalService
def evaluate_chunk_level(tool: RetrievalService, sample: EvalSample) -> dict:
    results = tool.search(sample.query)

    tp = sum(1 for r in results if is_relevant(r["text"], sample.relevant_kw))
    fp = len(results) - tp
    fn = sample.fn_count

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    scores = [r["score"] for r in results]

    if sample.act:
        wrong_act_chunks   = sum(1 for r in results if r.get("source", "") != sample.act)
        contamination_rate = wrong_act_chunks / len(results) if results else 0.0
    else:
        contamination_rate = 0.0

    return {
        "query":              sample.query,
        "act":                sample.act,
        "retrieved":          len(results),
        "tp": tp, "fp": fp, "fn": fn,
        "precision":          round(precision, 4),
        "recall":             round(recall, 4),
        "f1":                 round(f1, 4),
        "accuracy":           round(accuracy, 4),
        "mean_score":         round(sum(scores) / len(scores), 4) if scores else 0.0,
        "max_score":          round(max(scores), 4) if scores else 0.0,
        "min_score":          round(min(scores), 4) if scores else 0.0,
        "contamination_rate": round(contamination_rate, 4),
        "sources":            list({r["source"] for r in results}),
    }


def print_chunk_report(rows: list[dict], label: str = "CHAT") -> dict:
    sep = "-" * 100
    print(f"\n{color(f'CHUNK-LEVEL EVALUATION — {label} (Precision / Recall / F1 / Contamination)', 'bold')}")
    print(sep)
    hdr = (f"{'Query':<42} {'P':>6} {'R':>6} {'F1':>6} {'Acc':>6} "
           f"{'AvgSim':>8} {'Contam%':>9} {'TP/FP/FN':>10}")
    print(color(hdr, "cyan"))
    print(sep)

    for r in rows:
        q      = r["query"][:41]
        tpfpfn = f"{r['tp']}/{r['fp']}/{r['fn']}"
        contam = color(f"{r['contamination_rate']*100:.1f}%",
                       "red" if r["contamination_rate"] > 0.3 else "green")
        print(
            f"{q:<42} {grade(r['precision']):>14} {grade(r['recall']):>14} "
            f"{grade(r['f1']):>14} {grade(r['accuracy']):>14} "
            f"{r['mean_score']:>8.3f} {contam:>17} {tpfpfn:>10}"
        )

    print(sep)
    macro_p    = sum(r["precision"]          for r in rows) / len(rows)
    macro_r    = sum(r["recall"]             for r in rows) / len(rows)
    macro_f1   = sum(r["f1"]                 for r in rows) / len(rows)
    macro_acc  = sum(r["accuracy"]           for r in rows) / len(rows)
    macro_sim  = sum(r["mean_score"]         for r in rows) / len(rows)
    macro_cont = sum(r["contamination_rate"] for r in rows) / len(rows)

    print(
        f"{'MACRO AVERAGE':<42} {grade(macro_p):>14} {grade(macro_r):>14} "
        f"{grade(macro_f1):>14} {grade(macro_acc):>14} {macro_sim:>8.3f} "
        f"{macro_cont*100:>8.1f}%"
    )
    print(sep)
    print(f"\n{color('Thresholds: green >= 0.80  |  yellow >= 0.60  |  red < 0.60', 'cyan')}\n")

    if macro_p > macro_r + 0.15:
        print(color("Diagnosis: High precision but low recall.", "yellow"))
        print("  → Lower SCORE_THRESHOLD or raise TOP_K_RESULTS in .env\n")
    elif macro_r > macro_p + 0.15:
        print(color("Diagnosis: High recall but low precision.", "yellow"))
        print("  → Raise SCORE_THRESHOLD to filter noisy chunks\n")
    elif macro_cont > 0.25:
        print(color("Diagnosis: High cross-document contamination.", "red"))
        print("  → Add act-level metadata filtering in RetrievalService.search()\n")
    elif macro_f1 < 0.60:
        print(color("Diagnosis: Both precision and recall are low.", "red"))
        print("  → Fix chunking strategy (RecursiveCharacterTextSplitter, size=800, overlap=150)\n")
        print("  → Re-ingest with richer metadata (act_name, chunk_index)\n")
    else:
        print(color("Diagnosis: Retrieval quality looks healthy.", "green"))
        print("  → Run --mode ragas to evaluate answer quality\n")

    return {
        "macro_p": macro_p, "macro_r": macro_r,
        "macro_f1": macro_f1, "macro_acc": macro_acc, "macro_cont": macro_cont,
    }


# =============================================================================
# 4. CHUNK-LEVEL — ACT COMPARISON (dual-leg)
# =============================================================================

# ✅ updated type hint: LegalSearchTool → RetrievalService
def evaluate_chunk_level_compare(tool: RetrievalService, sample: CompareEvalSample) -> dict:
    def _eval_leg(query: str, kw: list[str], expected_act: str, fn_count: int) -> dict:
        results   = tool.search(query)
        tp        = sum(1 for r in results if is_relevant(r["text"], kw))
        fp        = len(results) - tp
        fn        = fn_count
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        scores    = [r["score"] for r in results]

        act_kw = expected_act.lower().split()
        wrong  = sum(
            1 for r in results
            if not any(kw in r.get("source", "").lower() for kw in act_kw)
        )
        contamination_rate = wrong / len(results) if results else 0.0

        return {
            "retrieved": len(results),
            "tp": tp, "fp": fp, "fn": fn,
            "precision":          round(precision, 4),
            "recall":             round(recall, 4),
            "f1":                 round(f1, 4),
            "mean_score":         round(sum(scores) / len(scores), 4) if scores else 0.0,
            "contamination_rate": round(contamination_rate, 4),
            "sources":            list({r["source"] for r in results}),
        }

    leg_a = _eval_leg(f"{sample.topic} {sample.act_a}", sample.relevant_kw_a,
                      sample.act_a, sample.fn_count_a)
    leg_b = _eval_leg(f"{sample.topic} {sample.act_b}", sample.relevant_kw_b,
                      sample.act_b, sample.fn_count_b)

    return {
        "topic":         sample.topic,
        "act_a":         sample.act_a,
        "act_b":         sample.act_b,
        "leg_a":         leg_a,
        "leg_b":         leg_b,
        "avg_f1":        round((leg_a["f1"]        + leg_b["f1"])        / 2, 4),
        "avg_precision": round((leg_a["precision"] + leg_b["precision"]) / 2, 4),
        "avg_recall":    round((leg_a["recall"]    + leg_b["recall"])    / 2, 4),
        "avg_contam":    round((leg_a["contamination_rate"] +
                                leg_b["contamination_rate"]) / 2, 4),
    }


def print_compare_chunk_report(rows: list[dict]) -> dict:
    sep = "-" * 110
    print(f"\n{color('CHUNK-LEVEL EVALUATION — ACT COMPARISON (dual-leg retrieval)', 'bold')}")
    print(sep)
    hdr = (f"{'Topic':<30} {'Act':<32} {'P':>6} {'R':>6} {'F1':>6} "
           f"{'AvgSim':>8} {'Contam%':>9} {'TP/FP/FN':>10}")
    print(color(hdr, "cyan"))
    print(sep)

    for r in rows:
        topic = r["topic"][:29]
        for leg_key, act_name in [("leg_a", r["act_a"]), ("leg_b", r["act_b"])]:
            leg    = r[leg_key]
            act    = act_name[:31]
            tpfpfn = f"{leg['tp']}/{leg['fp']}/{leg['fn']}"
            contam = color(
                f"{leg['contamination_rate']*100:.1f}%",
                "red" if leg["contamination_rate"] > 0.3 else "green"
            )
            print(
                f"{topic:<30} {act:<32} {grade(leg['precision']):>14} "
                f"{grade(leg['recall']):>14} {grade(leg['f1']):>14} "
                f"{leg['mean_score']:>8.3f} {contam:>17} {tpfpfn:>10}"
            )
            topic = ""

        avg_contam_str = color(
            f"{r['avg_contam']*100:.1f}%",
            "red" if r["avg_contam"] > 0.3 else "green"
        )
        print(
            f"  {'↳ Dual-leg avg':<60} {grade(r['avg_precision']):>14} "
            f"{grade(r['avg_recall']):>14} {grade(r['avg_f1']):>14} "
            f"{'':>8} {avg_contam_str:>17}"
        )
        print()

    print(sep)
    macro_p    = sum(r["avg_precision"] for r in rows) / len(rows)
    macro_r    = sum(r["avg_recall"]    for r in rows) / len(rows)
    macro_f1   = sum(r["avg_f1"]        for r in rows) / len(rows)
    macro_cont = sum(r["avg_contam"]    for r in rows) / len(rows)

    print(
        f"{'MACRO AVERAGE (dual-leg)':<64} {grade(macro_p):>14} "
        f"{grade(macro_r):>14} {grade(macro_f1):>14} "
        f"{'':>8} {macro_cont*100:>8.1f}%"
    )
    print(sep)
    print(f"\n{color('Thresholds: green >= 0.80  |  yellow >= 0.60  |  red < 0.60', 'cyan')}\n")

    if macro_cont > 0.30:
        print(color("Diagnosis: High cross-act contamination in comparison retrieval.", "red"))
        print("  → The keyword filter in _search_for_act() is too loose.")
        print("  → Consider adding act-level metadata to Qdrant payloads for exact filtering.\n")
    elif macro_f1 < 0.60:
        print(color("Diagnosis: Low F1 on both legs — retrieval misses relevant chunks.", "red"))
        print("  → Lower SCORE_THRESHOLD or raise TOP_K_RESULTS in .env\n")
    elif macro_p > macro_r + 0.15:
        print(color("Diagnosis: Good precision but low recall on comparison queries.", "yellow"))
        print("  → Raise TOP_K_RESULTS so more chunks per act are fetched.\n")
    else:
        print(color("Diagnosis: Comparison retrieval quality looks healthy.", "green"))
        print("  → Run --mode ragas_compare to evaluate structured output quality.\n")

    return {
        "macro_p": macro_p, "macro_r": macro_r,
        "macro_f1": macro_f1, "macro_cont": macro_cont,
    }


# =============================================================================
# 5. RAGAS — CHAT SERVICE
# =============================================================================

# ✅ updated type hint: LegalSearchTool → RetrievalService
def evaluate_ragas_chat(tool: RetrievalService, samples: list[EvalSample]) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_mistralai import ChatMistralAI
        from langchain_huggingface import HuggingFaceEmbeddings
        from app.tools.chat_tool import ChatTool  # ✅ use ChatTool instead of ChatService
    except ImportError as e:
        print(color(f"  Missing dependency: {e}", "red"))
        return {}

    print(f"\n{color('RAGAS — CHAT SERVICE  (Mistral judge + HuggingFace embeddings)', 'bold')}")
    print("  Building dataset...\n")

    mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
    llm = LangchainLLMWrapper(ChatMistralAI(model="mistral-small-2506",
                                             mistral_api_key=mistral_api_key))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"))

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    for m in metrics:
        m.llm = llm; m.embeddings = embeddings

    # ✅ use ChatTool instead of ChatService
    chat_tool = ChatTool()
    data    = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    DELAY   = float(os.getenv("EVAL_REQUEST_DELAY", "6.0"))

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample.query[:65]}...")
        results  = tool.search(sample.query)
        contexts = [r["text"] for r in results]
        answer   = _call_with_retry(
            lambda: asyncio.get_event_loop().run_until_complete(
                chat_tool.ask(sample.query))["answer"])  # ✅ chat_tool.ask()
        if answer is None:
            print(color("    Skipping — could not get answer.", "red")); continue
        data["question"].append(sample.query)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(sample.ground_truth)
        if i < len(samples) - 1:
            print(f"    Waiting {DELAY:.0f}s..."); time.sleep(DELAY)

    return _run_ragas_and_print(data, metrics, llm, embeddings, label="CHAT")


# =============================================================================
# 6. RAGAS — ACT COMPARISON SERVICE
# =============================================================================

# ✅ updated type hint: LegalSearchTool → RetrievalService
def evaluate_ragas_compare(tool: RetrievalService, samples: list[CompareEvalSample]) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_mistralai import ChatMistralAI
        from langchain_huggingface import HuggingFaceEmbeddings
        from app.tools.act_comparison_tool import ActComparisonTool
        from app.services.comparison_service import ComparisonService
    except ImportError as e:
        print(color(f"  Missing dependency: {e}", "red"))
        return {}

    print(f"\n{color('RAGAS — ACT COMPARISON SERVICE  (structured dual-act comparison)', 'bold')}")
    print("  Building dataset...\n")

    mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
    llm = LangchainLLMWrapper(ChatMistralAI(model="mistral-small-2506",
                                             mistral_api_key=mistral_api_key))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"))

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    for m in metrics:
        m.llm = llm; m.embeddings = embeddings

    act_tool = ActComparisonTool()
    service  = ComparisonService()
    data     = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    DELAY    = float(os.getenv("EVAL_REQUEST_DELAY", "6.0"))

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample.topic} | {sample.act_a} vs {sample.act_b}...")

        results_a = service._search_for_act(sample.topic, sample.act_a)
        results_b = service._search_for_act(sample.topic, sample.act_b)
        contexts  = [r["text"] for r in results_a] + [r["text"] for r in results_b]

        # ✅ use ActComparisonTool instead of ComparisonService directly
        answer = _call_with_retry(
            lambda s=sample: asyncio.get_event_loop().run_until_complete(
                act_tool.compare(s.topic, s.act_a, s.act_b)
            )["comparison"]
        )
        if answer is None:
            print(color("    Skipping — could not get comparison.", "red")); continue

        question = f"Compare {sample.act_a} and {sample.act_b} on the topic: {sample.topic}"
        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(sample.ground_truth)

        if i < len(samples) - 1:
            print(f"    Waiting {DELAY:.0f}s..."); time.sleep(DELAY)

    result = _run_ragas_and_print(data, metrics, llm, embeddings, label="COMPARE")

    if result.get("faithfulness", 1.0) < 0.85:
        print(color(
            "⚠  Comparison faithfulness low — Mistral may be inventing provisions "
            "not present in the retrieved chunks.", "red"
        ))
        print("  → Tighten the comparison prompt: reinforce 'Do not invent provisions'.\n")

    return result


# =============================================================================
# 7. SHARED RAGAS HELPERS
# =============================================================================

def _call_with_retry(fn, max_retries: int = 3) -> str | None:
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            is_rate = "429" in str(exc) or "rate_limit" in str(exc).lower()
            if is_rate and attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                print(color(f"    Rate limited — cooling down {wait}s (attempt {attempt+1}/{max_retries})", "yellow"))
                time.sleep(wait)
            else:
                raise
    return None


def _run_ragas_and_print(data: dict, metrics, llm, embeddings, label: str) -> dict:
    from datasets import Dataset
    from ragas import evaluate

    if not data["question"]:
        print(color("  No samples processed — aborting.", "red"))
        return {}

    dataset     = Dataset.from_dict(data)
    print(f"\n  Running RAGAS for {label} ({len(data['question'])} samples)...\n")
    result      = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    scores_dict = result._repr_dict

    sep = "-" * 58
    print(f"\n{color(f'RAGAS RESULTS — {label}', 'bold')}")
    print(sep)

    metric_labels = {
        "faithfulness":      "Faithfulness       (no hallucination)  ",
        "answer_relevancy":  "Answer relevancy   (on-topic answer)   ",
        "context_recall":    "Context recall     (retrieval complete) ",
        "context_precision": "Context precision  (retrieval accurate) ",
    }

    out = {}
    for key, lbl in metric_labels.items():
        try:
            val = float(scores_dict[key])
            print(f"  {lbl}  {grade(val)}")
            out[key] = val
        except (KeyError, TypeError):
            print(f"  {lbl}  N/A")

    print(sep)

    faith     = out.get("faithfulness", 1.0)
    recall    = out.get("context_recall", 1.0)
    precision = out.get("context_precision", 1.0)

    if faith < 0.85:
        print(color("\n⚠  Low faithfulness — LLM is going beyond the retrieved context.", "red"))
    else:
        print(color("\n✓  Faithfulness healthy — answers grounded in retrieved context.\n", "green"))
    if recall < 0.80:
        print(color("⚠  Context recall low — retriever is missing relevant passages.", "red"))
    if precision < 0.75:
        print(color("⚠  Context precision low — too many irrelevant chunks retrieved.", "red"))

    return out


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Legal Chatbot Evaluator")
    parser.add_argument(
        "--mode",
        choices=["chunk", "ragas", "ragas_compare", "full"],
        default="chunk",
        help=(
            "chunk          → fast retrieval-level F1 for chat + comparison (default)\n"
            "ragas          → RAGAS for Chat service\n"
            "ragas_compare  → RAGAS for Act Comparison service\n"
            "full           → all of the above"
        ),
    )
    args = parser.parse_args()

    print(color("\n=== RAG Legal Advisor — Evaluation Suite ===\n", "bold"))
    print(f"  Embedding model  : {os.getenv('EMBEDDING_MODEL', 'see .env')}")
    print(f"  Score threshold  : {os.getenv('SCORE_THRESHOLD', '0.50')}")
    print(f"  Top-K results    : {os.getenv('TOP_K_RESULTS', '10')}")
    print(f"  RERANK_TOP_K     : {os.getenv('RERANK_TOP_K', '1')}")
    print(f"  Chat samples     : {len(CHAT_EVAL_DATASET)}")
    print(f"  Compare samples  : {len(COMPARE_EVAL_DATASET)}\n")

    # ✅ RetrievalService instead of LegalSearchTool
    tool        = RetrievalService()
    all_results = {}

    print(color("─── CHAT RETRIEVAL ───────────────────────────────────", "magenta"))
    all_results["chunk_chat"] = print_chunk_report(
        [evaluate_chunk_level(tool, s) for s in CHAT_EVAL_DATASET], label="CHAT")

    print(color("─── ACT COMPARISON RETRIEVAL (dual-leg) ──────────────", "magenta"))
    all_results["chunk_compare"] = print_compare_chunk_report(
        [evaluate_chunk_level_compare(tool, s) for s in COMPARE_EVAL_DATASET])

    if args.mode in ("ragas", "full"):
        all_results["ragas_chat"] = evaluate_ragas_chat(tool, CHAT_EVAL_DATASET)

    if args.mode in ("ragas_compare", "full"):
        all_results["ragas_compare"] = evaluate_ragas_compare(tool, COMPARE_EVAL_DATASET)

    save_results(all_results, args.mode)
    print(color("\n=== Evaluation complete ===\n", "bold"))


if __name__ == "__main__":
    main()