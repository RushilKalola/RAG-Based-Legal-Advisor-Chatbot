"""
RAG Evaluation Script — F1, Precision, Recall, ROUGE, BERTScore, RAGAS
=======================================================================
Place in: RAG-Based-Legal-Advisor-Chatbot/eval.py
Run:      python eval.py
          python eval.py --mode rouge
          python eval.py --mode bertscore
          python eval.py --mode ragas
          python eval.py --mode full

Install deps:
    pip install rouge-score bert-score ragas==0.2.15 datasets langchain-mistralai==0.2.10 langchain-community
"""

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass

# ── Adjust path so app/ is importable ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from app.tools.legal_search_tool import LegalSearchTool


# ═════════════════════════════════════════════════════════════════════════════
# 1. GROUND-TRUTH DATASET
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalSample:
    query: str
    relevant_kw: list[str]
    ground_truth: str
    fn_count: int = 0


EVAL_DATASET: list[EvalSample] = [
    EvalSample(
        query="What is the punishment for murder under Bharatiya Nyaya Sanhita?",
        relevant_kw=["murder", "death", "imprisonment for life", "302", "culpable homicide"],
        ground_truth=(
            "Under the Bharatiya Nyaya Sanhita, whoever commits murder shall be "
            "punished with death or imprisonment for life and shall also be liable "
            "to fine."
        ),
        fn_count=1,
    ),
    EvalSample(
        query="Rights of an accused during trial under Code of Criminal Procedure",
        relevant_kw=["fair trial", "accused", "defence", "legal aid", "bail", "right"],
        ground_truth=(
            "An accused person has the right to a fair trial, right to be represented "
            "by a lawyer, right to bail in bailable offences, and right to be informed "
            "of charges under the Code of Criminal Procedure."
        ),
        fn_count=2,
    ),
    EvalSample(
        query="How to file a consumer complaint under Consumer Protection Act 2019?",
        relevant_kw=["complaint", "consumer", "redressal", "commission", "filing", "district"],
        ground_truth=(
            "A consumer complaint can be filed before the District Consumer Disputes "
            "Redressal Commission for claims up to one crore rupees. The complaint must "
            "be filed within two years of the cause of action."
        ),
        fn_count=1,
    ),
    EvalSample(
        query="Penalties for hacking and unauthorized access under IT Act 2000",
        relevant_kw=["hacking", "unauthorised access", "section 43", "penalty", "computer", "damage"],
        ground_truth=(
            "Section 43 of the Information Technology Act 2000 prescribes penalty and "
            "compensation for unauthorized access, damage to computer systems, and "
            "downloading or copying data without permission."
        ),
        fn_count=0,
    ),
    EvalSample(
        query="Director liability under Companies Act 2013",
        relevant_kw=["director", "liability", "board", "company", "fiduciary", "duty"],
        ground_truth=(
            "Directors under the Companies Act 2013 have fiduciary duties towards the "
            "company, its shareholders, and the public. They can be held personally "
            "liable for fraudulent or wrongful acts committed in the name of the company."
        ),
        fn_count=1,
    ),
    EvalSample(
        query="Fundamental rights guaranteed under Constitution of India",
        relevant_kw=["fundamental rights", "article 14", "article 19", "article 21", "equality", "freedom"],
        ground_truth=(
            "Part III of the Constitution of India guarantees fundamental rights including "
            "right to equality (Article 14), right to freedom (Article 19), right to life "
            "and personal liberty (Article 21), and right to constitutional remedies (Article 32)."
        ),
        fn_count=2,
    ),
    EvalSample(
        query="Motor vehicle accident compensation claim procedure",
        relevant_kw=["accident", "compensation", "motor vehicle", "tribunal", "claim", "insurance"],
        ground_truth=(
            "Claims for motor vehicle accident compensation are filed before the Motor "
            "Accidents Claims Tribunal. The claimant must prove negligence and the amount "
            "of loss suffered. Insurance companies are liable to pay the awarded compensation."
        ),
        fn_count=1,
    ),
    EvalSample(
        query="Rules of evidence admissibility under Indian Evidence Act",
        relevant_kw=["evidence", "admissible", "relevance", "confession", "witness", "document"],
        ground_truth=(
            "Under the Indian Evidence Act, evidence must be relevant to the facts in issue "
            "to be admissible. Confessions made to police officers are generally not admissible. "
            "Documentary evidence must be proved by primary or secondary evidence."
        ),
        fn_count=1,
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def color(text: str, code: str) -> str:
    codes = {
        "green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m",
        "bold": "\033[1m", "reset": "\033[0m", "cyan": "\033[96m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"


def grade(val: float) -> str:
    if val >= 0.75:
        return color(f"{val:.3f}", "green")
    if val >= 0.55:
        return color(f"{val:.3f}", "yellow")
    return color(f"{val:.3f}", "red")


# ═════════════════════════════════════════════════════════════════════════════
# 3. CHUNK-LEVEL EVALUATION  (Precision / Recall / F1)
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_chunk_level(tool: LegalSearchTool, sample: EvalSample) -> dict:
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

    return {
        "query":       sample.query,
        "retrieved":   len(results),
        "tp": tp, "fp": fp, "fn": fn,
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "f1":          round(f1, 4),
        "accuracy":    round(accuracy, 4),
        "mean_score":  round(sum(scores) / len(scores), 4) if scores else 0.0,
        "max_score":   round(max(scores), 4) if scores else 0.0,
        "min_score":   round(min(scores), 4) if scores else 0.0,
        "sources":     list({r["source"] for r in results}),
        "results_raw": results,
    }


def print_chunk_report(rows: list[dict]) -> dict:
    sep = "-" * 90
    print(f"\n{color('CHUNK-LEVEL EVALUATION (Precision / Recall / F1)', 'bold')}")
    print(sep)
    hdr = f"{'Query':<42} {'P':>6} {'R':>6} {'F1':>6} {'Acc':>6} {'AvgSim':>8} {'TP/FP/FN':>10}"
    print(color(hdr, "cyan"))
    print(sep)

    for r in rows:
        q    = r["query"][:41]
        tpfp = f"{r['tp']}/{r['fp']}/{r['fn']}"
        print(f"{q:<42} {grade(r['precision']):>14} {grade(r['recall']):>14} "
              f"{grade(r['f1']):>14} {grade(r['accuracy']):>14} "
              f"{r['mean_score']:>8.3f} {tpfp:>10}")

    print(sep)
    macro_p   = sum(r["precision"]  for r in rows) / len(rows)
    macro_r   = sum(r["recall"]     for r in rows) / len(rows)
    macro_f1  = sum(r["f1"]         for r in rows) / len(rows)
    macro_acc = sum(r["accuracy"]   for r in rows) / len(rows)
    macro_sim = sum(r["mean_score"] for r in rows) / len(rows)

    print(f"{'MACRO AVERAGE':<42} {grade(macro_p):>14} {grade(macro_r):>14} "
          f"{grade(macro_f1):>14} {grade(macro_acc):>14} {macro_sim:>8.3f}")
    print(sep)
    print(f"\n{color('Thresholds: green >= 0.75  |  yellow >= 0.55  |  red < 0.55', 'cyan')}\n")

    if macro_p > macro_r + 0.15:
        print(color("Diagnosis: High precision but low recall.", "yellow"))
        print("  → Lower SCORE_THRESHOLD or raise TOP_K_RESULTS in .env\n")
    elif macro_r > macro_p + 0.15:
        print(color("Diagnosis: High recall but low precision.", "yellow"))
        print("  → Raise SCORE_THRESHOLD to filter noisy chunks\n")
    elif macro_f1 < 0.55:
        print(color("Diagnosis: Both precision and recall are low.", "red"))
        print("  → Consider a larger embedding model (e.g. all-mpnet-base-v2)\n")
        print("  → Try chunk_size=500-800 in .env\n")
    else:
        print(color("Diagnosis: Retrieval quality looks healthy.", "green"))
        print("  → Focus on answer quality with RAGAS faithfulness scoring\n")

    return {"macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1}


# # ═════════════════════════════════════════════════════════════════════════════
# # 4. ROUGE  (Mistral LLM answer vs ground truth)
# # ═════════════════════════════════════════════════════════════════════════════

# def evaluate_rouge(samples: list[EvalSample]):
#     try:
#         from rouge_score import rouge_scorer
#     except ImportError:
#         print(color("  rouge-score not installed. Run: pip install rouge-score", "red"))
#         return

#     try:
#         from app.services.chat_services import ChatService
#     except ImportError:
#         print(color("  Could not import ChatService.", "red"))
#         return

#     print(f"\n{color('ROUGE SCORES  (Mistral answer vs ground truth)', 'bold')}")
#     service = ChatService()
#     scorer  = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

#     sep = "-" * 70
#     print(sep)
#     print(color(f"{'Query':<42} {'R-1':>7} {'R-2':>7} {'R-L':>7}", "cyan"))
#     print(sep)

#     totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

#     for sample in samples:
#         print(f"  Querying Mistral for: {sample.query[:50]}...")
#         answer = asyncio.run(service.get_answer(sample.query))["answer"]
#         scores = scorer.score(sample.ground_truth, answer)
#         r1 = scores["rouge1"].fmeasure
#         r2 = scores["rouge2"].fmeasure
#         rl = scores["rougeL"].fmeasure
#         totals["rouge1"] += r1
#         totals["rouge2"] += r2
#         totals["rougeL"] += rl
#         print(f"{sample.query[:41]:<42} {grade(r1):>15} {grade(r2):>15} {grade(rl):>15}")

#     n = len(samples)
#     print(sep)
#     print(f"{'MACRO AVERAGE':<42} "
#           f"{grade(totals['rouge1']/n):>15} "
#           f"{grade(totals['rouge2']/n):>15} "
#           f"{grade(totals['rougeL']/n):>15}")
#     print(sep)


# # ═════════════════════════════════════════════════════════════════════════════
# # 5. BERTScore  (semantic similarity of Mistral answer vs ground truth)
# # ═════════════════════════════════════════════════════════════════════════════

# def evaluate_bertscore(samples: list[EvalSample]):
#     try:
#         from bert_score import score as bert_score
#     except ImportError:
#         print(color("  bert-score not installed. Run: pip install bert-score", "red"))
#         return

#     try:
#         from app.services.chat_services import ChatService
#     except ImportError:
#         print(color("  Could not import ChatService.", "red"))
#         return

#     print(f"\n{color('BERTScore  (semantic similarity of Mistral answer vs ground truth)', 'bold')}")
#     service    = ChatService()
#     candidates = []

#     for sample in samples:
#         print(f"  Querying Mistral for: {sample.query[:50]}...")
#         answer = asyncio.run(service.get_answer(sample.query))["answer"]
#         candidates.append(answer)

#     references = [s.ground_truth for s in samples]
#     P, R, F1   = bert_score(candidates, references, lang="en", verbose=False)

#     sep = "-" * 60
#     print(sep)
#     print(color(f"{'Query':<42} {'F1':>8}", "cyan"))
#     print(sep)
#     for sample, f in zip(samples, F1.tolist()):
#         print(f"{sample.query[:41]:<42} {grade(f):>16}")
#     print(sep)
#     avg = sum(F1.tolist()) / len(F1)
#     print(f"{'MACRO AVERAGE':<42} {grade(avg):>16}")
#     print(sep)


# ═════════════════════════════════════════════════════════════════════════════
# 6. RAGAS  (faithfulness, answer relevancy, context recall, context precision)
#    — uses Mistral as LLM judge + all-mpnet-base-v2 for embeddings
#    — no OpenAI key required
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_ragas(tool: LegalSearchTool, samples: list[EvalSample]):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_mistralai import ChatMistralAI
        from langchain_huggingface import HuggingFaceEmbeddings
        from app.services.chat_services import ChatService
    except ImportError as e:
        print(color(f"  Missing dependency: {e}", "red"))
        print("  Run: pip install ragas==0.2.15 langchain-mistralai langchain-community datasets")
        return

    print(f"\n{color('RAGAS EVALUATION  (Mistral as LLM judge — no OpenAI needed)', 'bold')}")
    print("  Building dataset...\n")

    # ── LLM judge: Mistral ────────────────────────────────────────────────
    mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
    llm = LangchainLLMWrapper(ChatMistralAI(
        model="mistral-small-2506",
        mistral_api_key=mistral_api_key,
    ))

    # ── Embeddings: HuggingFace — no OpenAI key needed ────────────────────
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    ))

    # ── Assign LLM + embeddings to every metric ───────────────────────────
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    for metric in metrics:
        metric.llm        = llm
        metric.embeddings = embeddings

    # ── Build dataset ─────────────────────────────────────────────────────
    import time as _time

    service = ChatService()
    data    = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    # 3-second gap between requests keeps us well under Mistral's free-tier
    # rate limit (~20 req/min).  The retry logic inside ChatService handles
    # any occasional burst that still sneaks through.
    INTER_REQUEST_DELAY = 3.0   # seconds — reduce to 1.0 on a paid plan

    for i, sample in enumerate(samples):
        print(f"  Processing: {sample.query[:60]}...")
        results  = tool.search(sample.query)
        contexts = [r["text"] for r in results]

        # get_answer already retries internally on 429; this outer loop is a
        # last-resort safety net in case all retries inside are exhausted.
        answer = None
        for outer_attempt in range(3):
            try:
                answer = asyncio.run(service.get_answer(sample.query))["answer"]
                break
            except Exception as exc:
                if "429" in str(exc) or "rate_limit" in str(exc).lower():
                    wait = 60 * (outer_attempt + 1)   # 60s, 120s, 180s
                    print(
                        f"    [eval] Still rate-limited after internal retries. "
                        f"Cooling down for {wait}s… (outer attempt {outer_attempt + 1}/3)"
                    )
                    _time.sleep(wait)
                else:
                    raise

        if answer is None:
            print(f"    [eval] Skipping sample — could not get answer after all retries.")
            continue

        data["question"].append(sample.query)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(sample.ground_truth)

        # Polite delay between every call (skip after the last sample)
        if i < len(samples) - 1:
            print(f"    Waiting {INTER_REQUEST_DELAY:.0f}s before next request...")
            _time.sleep(INTER_REQUEST_DELAY)

    if not data["question"]:
        print("  No samples processed — aborting RAGAS evaluation.")
        return

    dataset = Dataset.from_dict(data)

    # ── Run RAGAS ─────────────────────────────────────────────────────────
    print("\n  Running RAGAS evaluation (this may take 1-2 minutes)...\n")
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    scores_dict = result._repr_dict

    sep = "-" * 55
    print(f"\n{color('RAGAS RESULTS', 'bold')}")
    print(sep)

    metric_labels = {
        "faithfulness":      "Faithfulness       (no hallucination)",
        "answer_relevancy":  "Answer relevancy   (answers the question)",
        "context_recall":    "Context recall     (retrieval completeness)",
        "context_precision": "Context precision  (retrieval accuracy)",
    }

    for key, label in metric_labels.items():
        try:
            val = float(scores_dict[key])
            print(f"  {label:<45} {grade(val)}")
        except (KeyError, TypeError):
            print(f"  {label:<45} N/A")

    print(sep)

    # ── Diagnosis ─────────────────────────────────────────────────────────
    try:
        faith = float(scores_dict["faithfulness"])
    except (KeyError, TypeError):
        faith = 1.0

    if faith < 0.70:
        print(color("\nWarning: Low faithfulness — Mistral is hallucinating legal info.", "red"))
        print("  → Add a stricter system prompt in chat_services.py:")
        print("    'Answer ONLY from the provided context. Do not add external knowledge.'\n")
    else:
        print(color("\nFaithfulness is healthy — answers are grounded in retrieved context.\n", "green"))


# ═════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAG Legal Chatbot Evaluator")
    parser.add_argument(
        "--mode",
        choices=["chunk", "rouge", "bertscore", "ragas", "full"],
        default="chunk",
        help="Evaluation mode (default: chunk)",
    )
    args = parser.parse_args()

    print(color("\n=== RAG Legal Advisor — Evaluation Suite ===\n", "bold"))
    print(f"  Embedding model : {os.getenv('EMBEDDING_MODEL', 'see .env')}")
    print(f"  Score threshold : {os.getenv('SCORE_THRESHOLD', '0.55')}")
    print(f"  Top-K results   : {os.getenv('TOP_K_RESULTS', '5')}")
    print(f"  Chunk size      : {os.getenv('CHUNK_SIZE', '500')}")
    print(f"  Queries         : {len(EVAL_DATASET)}\n")

    tool = LegalSearchTool()

    # Always run chunk-level (fast, no extra deps)
    rows = [evaluate_chunk_level(tool, s) for s in EVAL_DATASET]
    print_chunk_report(rows)

    # if args.mode in ("rouge", "full"):
    #     evaluate_rouge(EVAL_DATASET)

    # if args.mode in ("bertscore", "full"):
    #     evaluate_bertscore(EVAL_DATASET)

    if args.mode in ("ragas", "full"):
        evaluate_ragas(tool, EVAL_DATASET)

    print(color("\n=== Evaluation complete ===\n", "bold"))


if __name__ == "__main__":
    main()