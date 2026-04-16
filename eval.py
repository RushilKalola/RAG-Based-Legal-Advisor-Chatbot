"""
RAG Evaluation Script — Chunk-level F1 + RAGAS (Chat & Section)
================================================================
Place in: RAG-Based-Legal-Advisor-Chatbot/eval.py

Run:
    python eval.py                     # chunk-level only (fast)
    python eval.py --mode ragas        # RAGAS for chat
    python eval.py --mode ragas_section  # RAGAS for section lookup
    python eval.py --mode full         # everything

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
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# ── Adjust path so app/ is importable ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from app.tools.legal_search_tool import LegalSearchTool


# ═════════════════════════════════════════════════════════════════════════════
# 1. EVAL DATASETS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalSample:
    query: str
    relevant_kw: list[str]
    ground_truth: str
    fn_count: int = 0          # known false negatives (missed relevant chunks)
    act: str = ""              # which PDF this should come from


# ── Chat eval dataset (open-ended legal questions) ────────────────────────────
CHAT_EVAL_DATASET: list[EvalSample] = [
    EvalSample(
        query="What is the punishment for murder under Bharatiya Nyaya Sanhita?",
        relevant_kw=["murder", "death", "imprisonment for life", "culpable homicide"],
        ground_truth=(
            "Under the Bharatiya Nyaya Sanhita, whoever commits murder shall be "
            "punished with death or imprisonment for life and shall also be liable to fine."
        ),
        fn_count=1,
        act="Bharatiya Nyaya Sanhita.pdf",
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
        act="Code of Criminal Procedure.pdf",
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
        act="Consumer Protection Act 2019.pdf",
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
        act="Information Technology Act 2000.pdf",
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
        act="Companies Act 2013.pdf",
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
        act="Constitution of India.pdf",
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
        act="The Motor Vehicles Act.pdf",
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
        act="Indian Evidence Act.pdf",
    ),
    # ── Additional samples for better statistical reliability ─────────────────
    EvalSample(
        query="What is the procedure for filing a civil suit under Code of Civil Procedure?",
        relevant_kw=["civil suit", "plaint", "court", "jurisdiction", "filing", "order"],
        ground_truth=(
            "A civil suit is filed by presenting a plaint to the court of competent jurisdiction. "
            "The plaint must contain facts constituting the cause of action and the relief claimed."
        ),
        fn_count=1,
        act="Code of Civil Procedure.pdf",
    ),
    EvalSample(
        query="What are the data protection obligations under IT Act 2000?",
        relevant_kw=["data", "protection", "sensitive", "personal information", "body corporate", "reasonable security"],
        ground_truth=(
            "Section 43A of the IT Act requires body corporates handling sensitive personal data "
            "to maintain reasonable security practices. Failure to do so makes them liable to pay "
            "compensation to affected persons."
        ),
        fn_count=1,
        act="Information Technology Act 2000.pdf",
    ),
    EvalSample(
        query="Rights of shareholders under Companies Act 2013",
        relevant_kw=["shareholder", "dividend", "voting", "annual general meeting", "rights", "equity"],
        ground_truth=(
            "Shareholders under the Companies Act 2013 have the right to receive dividends, "
            "vote at general meetings, inspect statutory registers, and receive annual reports. "
            "They can also approach the tribunal for relief against oppression and mismanagement."
        ),
        fn_count=1,
        act="Companies Act 2013.pdf",
    ),
    EvalSample(
        query="Bail provisions for non-bailable offences under CrPC",
        relevant_kw=["bail", "non-bailable", "court", "discretion", "session", "high court"],
        ground_truth=(
            "For non-bailable offences, bail is at the discretion of the court. The Sessions Court "
            "or High Court may grant bail considering factors like nature of offence, evidence, "
            "and likelihood of fleeing."
        ),
        fn_count=1,
        act="Code of Criminal Procedure.pdf",
    ),
]


# ── Section eval dataset (verbatim section lookup) ────────────────────────────
SECTION_EVAL_DATASET: list[EvalSample] = [
    EvalSample(
        query="Article 21 Constitution of India",
        relevant_kw=["life", "personal liberty", "procedure established by law"],
        ground_truth=(
            "No person shall be deprived of his life or personal liberty except "
            "according to procedure established by law."
        ),
        fn_count=0,
        act="Constitution of India.pdf",
    ),
    EvalSample(
        query="Article 14 Constitution of India right to equality",
        relevant_kw=["equality", "equal protection", "law", "territory of india"],
        ground_truth=(
            "The State shall not deny to any person equality before the law or the "
            "equal protection of the laws within the territory of India."
        ),
        fn_count=0,
        act="Constitution of India.pdf",
    ),
    EvalSample(
        query="Section 43 Information Technology Act 2000 penalty",
        relevant_kw=["penalty", "computer", "unauthorised access", "damages", "section 43"],
        ground_truth=(
            "Section 43 of the IT Act prescribes that if any person without permission "
            "of the owner accesses or secures access to a computer, computer system or "
            "computer network, he shall be liable to pay damages by way of compensation."
        ),
        fn_count=0,
        act="Information Technology Act 2000.pdf",
    ),
    EvalSample(
        query="Section 302 Bharatiya Nyaya Sanhita punishment for murder",
        relevant_kw=["murder", "death", "imprisonment for life", "fine"],
        ground_truth=(
            "Whoever commits murder shall be punished with death or imprisonment for "
            "life and shall also be liable to fine."
        ),
        fn_count=0,
        act="Bharatiya Nyaya Sanhita.pdf",
    ),
    EvalSample(
        query="Section 166 Motor Vehicles Act application for compensation",
        relevant_kw=["compensation", "application", "claims tribunal", "accident", "motor vehicle"],
        ground_truth=(
            "An application for compensation arising out of an accident of the nature "
            "specified in sub-section (1) of section 165 may be made by the person who "
            "has sustained the injury or by the owner of the property."
        ),
        fn_count=0,
        act="The Motor Vehicles Act.pdf",
    ),
    EvalSample(
        query="Section 24 Consumer Protection Act 2019 jurisdiction of District Commission",
        relevant_kw=["district commission", "jurisdiction", "one crore", "complaint", "consumer"],
        ground_truth=(
            "The District Commission shall have jurisdiction to entertain complaints where "
            "the value of goods or services paid as consideration does not exceed one crore rupees."
        ),
        fn_count=0,
        act="Consumer Protection Act 2019.pdf",
    ),
    EvalSample(
        query="Section 3 Indian Evidence Act relevancy of facts",
        relevant_kw=["relevancy", "facts", "evidence", "issue", "relevant"],
        ground_truth=(
            "One fact is said to be relevant to another when the one is connected with "
            "the other in any of the ways referred to in the provisions of this Act "
            "relating to the relevancy of facts."
        ),
        fn_count=0,
        act="Indian Evidence Act.pdf",
    ),
    EvalSample(
        query="Section 149 Companies Act 2013 composition of board of directors",
        relevant_kw=["board", "directors", "company", "independent director", "composition"],
        ground_truth=(
            "Every company shall have a Board of Directors consisting of individuals as "
            "directors and shall have a minimum number of three directors in the case of "
            "a public company, two directors in the case of a private company."
        ),
        fn_count=0,
        act="Companies Act 2013.pdf",
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
    """Save results to eval_results/ folder for tracking improvement over time."""
    os.makedirs("eval_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = f"eval_results/{timestamp}_{mode}.json"
    results["timestamp"] = timestamp
    results["mode"] = mode
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(color(f"\n  Results saved → {filepath}", "cyan"))


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

    # Cross-document contamination check:
    # how many retrieved chunks came from a WRONG act?
    if sample.act:
        wrong_act_chunks = sum(
            1 for r in results
            if r.get("source", "") != sample.act
        )
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
        contam = color(f"{r['contamination_rate']*100:.1f}%", "red" if r["contamination_rate"] > 0.3 else "green")
        print(
            f"{q:<42} {grade(r['precision']):>14} {grade(r['recall']):>14} "
            f"{grade(r['f1']):>14} {grade(r['accuracy']):>14} "
            f"{r['mean_score']:>8.3f} {contam:>17} {tpfpfn:>10}"
        )

    print(sep)
    macro_p     = sum(r["precision"]          for r in rows) / len(rows)
    macro_r     = sum(r["recall"]             for r in rows) / len(rows)
    macro_f1    = sum(r["f1"]                 for r in rows) / len(rows)
    macro_acc   = sum(r["accuracy"]           for r in rows) / len(rows)
    macro_sim   = sum(r["mean_score"]         for r in rows) / len(rows)
    macro_cont  = sum(r["contamination_rate"] for r in rows) / len(rows)

    print(
        f"{'MACRO AVERAGE':<42} {grade(macro_p):>14} {grade(macro_r):>14} "
        f"{grade(macro_f1):>14} {grade(macro_acc):>14} {macro_sim:>8.3f} "
        f"{macro_cont*100:>8.1f}%"
    )
    print(sep)
    print(f"\n{color('Thresholds: green >= 0.80  |  yellow >= 0.60  |  red < 0.60', 'cyan')}\n")

    # ── Diagnosis ─────────────────────────────────────────────────────────
    if macro_p > macro_r + 0.15:
        print(color("Diagnosis: High precision but low recall.", "yellow"))
        print("  → Lower SCORE_THRESHOLD or raise TOP_K_RESULTS in .env\n")
    elif macro_r > macro_p + 0.15:
        print(color("Diagnosis: High recall but low precision.", "yellow"))
        print("  → Raise SCORE_THRESHOLD to filter noisy chunks\n")
    elif macro_cont > 0.25:
        print(color("Diagnosis: High cross-document contamination.", "red"))
        print("  → Add act-level metadata filtering in LegalSearchTool.search()\n")
    elif macro_f1 < 0.60:
        print(color("Diagnosis: Both precision and recall are low.", "red"))
        print("  → Fix chunking strategy (RecursiveCharacterTextSplitter, size=800, overlap=150)\n")
        print("  → Re-ingest with richer metadata (act_name, chunk_index)\n")
    else:
        print(color("Diagnosis: Retrieval quality looks healthy.", "green"))
        print("  → Run --mode ragas to evaluate answer quality\n")

    return {
        "macro_p":    macro_p,
        "macro_r":    macro_r,
        "macro_f1":   macro_f1,
        "macro_acc":  macro_acc,
        "macro_cont": macro_cont,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. RAGAS — CHAT SERVICE
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_ragas_chat(tool: LegalSearchTool, samples: list[EvalSample]) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_mistralai import ChatMistralAI
        from langchain_huggingface import HuggingFaceEmbeddings
        from app.services.chat_services import ChatService
    except ImportError as e:
        print(color(f"  Missing dependency: {e}", "red"))
        return {}

    print(f"\n{color('RAGAS — CHAT SERVICE  (Mistral judge + HuggingFace embeddings)', 'bold')}")
    print("  Building dataset...\n")

    mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
    llm = LangchainLLMWrapper(ChatMistralAI(
        model="mistral-small-2506",
        mistral_api_key=mistral_api_key,
    ))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    ))

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    for m in metrics:
        m.llm        = llm
        m.embeddings = embeddings

    service = ChatService()
    data    = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    DELAY   = float(os.getenv("EVAL_REQUEST_DELAY", "6.0"))

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample.query[:65]}...")
        results  = tool.search(sample.query)
        contexts = [r["text"] for r in results]

        answer = _call_with_retry(
            lambda: asyncio.get_event_loop().run_until_complete(
                service.get_answer(sample.query)
            )["answer"]
        )

        if answer is None:
            print(color(f"    Skipping — could not get answer.", "red"))
            continue

        data["question"].append(sample.query)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(sample.ground_truth)

        if i < len(samples) - 1:
            print(f"    Waiting {DELAY:.0f}s...")
            time.sleep(DELAY)

    return _run_ragas_and_print(data, metrics, llm, embeddings, label="CHAT")


# ═════════════════════════════════════════════════════════════════════════════
# 5. RAGAS — SECTION SERVICE  (new — was never evaluated before)
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_ragas_section(tool: LegalSearchTool, samples: list[EvalSample]) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_mistralai import ChatMistralAI
        from langchain_huggingface import HuggingFaceEmbeddings
        from app.services.section_services import SectionService
    except ImportError as e:
        print(color(f"  Missing dependency: {e}", "red"))
        return {}

    print(f"\n{color('RAGAS — SECTION SERVICE  (verbatim section extraction)', 'bold')}")
    print("  Building dataset...\n")

    mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
    llm = LangchainLLMWrapper(ChatMistralAI(
        model="mistral-small-2506",
        mistral_api_key=mistral_api_key,
    ))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    ))

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    for m in metrics:
        m.llm        = llm
        m.embeddings = embeddings

    service = SectionService()
    data    = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    DELAY   = float(os.getenv("EVAL_REQUEST_DELAY", "6.0"))

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample.query[:65]}...")
        results  = tool.search(sample.query)
        contexts = [r["text"] for r in results]

        # SectionService.get_section() is async — use event loop
        answer = _call_with_retry(
            lambda: asyncio.get_event_loop().run_until_complete(
                service.get_section(sample.query)
            )["answer"]
        )

        if answer is None:
            print(color(f"    Skipping — could not get answer.", "red"))
            continue

        data["question"].append(sample.query)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(sample.ground_truth)

        if i < len(samples) - 1:
            print(f"    Waiting {DELAY:.0f}s...")
            time.sleep(DELAY)

    return _run_ragas_and_print(data, metrics, llm, embeddings, label="SECTION")


# ═════════════════════════════════════════════════════════════════════════════
# 6. SHARED HELPERS FOR RAGAS
# ═════════════════════════════════════════════════════════════════════════════

def _call_with_retry(fn, max_retries: int = 3) -> str | None:
    """Call fn() with exponential backoff on rate-limit errors."""
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

    dataset = Dataset.from_dict(data)
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

    # ── Per-metric diagnosis ──────────────────────────────────────────────
    faith = out.get("faithfulness", 1.0)
    recall = out.get("context_recall", 1.0)
    precision = out.get("context_precision", 1.0)

    if faith < 0.85:
        print(color("\n⚠  Low faithfulness — LLM is going beyond the retrieved context.", "red"))
        print("   → Tighten the system prompt: 'Answer ONLY from the context below.'")
        print("   → Add a negative instruction: 'Do NOT use prior legal knowledge.'\n")
    else:
        print(color("\n✓  Faithfulness healthy — answers grounded in retrieved context.\n", "green"))

    if recall < 0.80:
        print(color("⚠  Context recall low — retriever is missing relevant passages.", "red"))
        print("   → Lower SCORE_THRESHOLD to 0.40 in .env")
        print("   → Raise TOP_K_RESULTS to 12")
        print("   → Fix chunking: RecursiveCharacterTextSplitter(size=800, overlap=150)\n")

    if precision < 0.75:
        print(color("⚠  Context precision low — too many irrelevant chunks retrieved.", "red"))
        print("   → Add act-level metadata filtering in LegalSearchTool")
        print("   → Re-ingest with 'act_name' payload field\n")

    if label == "SECTION" and faith < 0.90:
        print(color("⚠  Section service needs higher faithfulness (target ≥ 0.90).", "yellow"))
        print("   → Remove the 100-word cap from the section prompt")
        print("   → Prompt must say: 'Return the exact section text — do NOT paraphrase.'\n")

    return out


# ═════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAG Legal Chatbot Evaluator")
    parser.add_argument(
        "--mode",
        choices=["chunk", "ragas", "ragas_section", "full"],
        default="chunk",
        help=(
            "chunk          → fast retrieval-level F1 (default)\n"
            "ragas          → RAGAS for Chat service\n"
            "ragas_section  → RAGAS for Section service\n"
            "full           → all of the above"
        ),
    )
    args = parser.parse_args()

    print(color("\n=== RAG Legal Advisor — Evaluation Suite ===\n", "bold"))
    print(f"  Embedding model  : {os.getenv('EMBEDDING_MODEL', 'see .env')}")
    print(f"  Score threshold  : {os.getenv('SCORE_THRESHOLD', '0.50')}")
    print(f"  Top-K results    : {os.getenv('TOP_K_RESULTS', '10')}")
    print(f"  Chat samples     : {len(CHAT_EVAL_DATASET)}")
    print(f"  Section samples  : {len(SECTION_EVAL_DATASET)}\n")

    tool        = LegalSearchTool()
    all_results = {}

    # ── Always run chunk-level (fast, no extra deps) ──────────────────────
    print(color("─── CHAT RETRIEVAL ───────────────────────────────────", "magenta"))
    chat_rows = [evaluate_chunk_level(tool, s) for s in CHAT_EVAL_DATASET]
    chat_chunk = print_chunk_report(chat_rows, label="CHAT")
    all_results["chunk_chat"] = chat_chunk

    print(color("─── SECTION RETRIEVAL ────────────────────────────────", "magenta"))
    sec_rows = [evaluate_chunk_level(tool, s) for s in SECTION_EVAL_DATASET]
    sec_chunk = print_chunk_report(sec_rows, label="SECTION")
    all_results["chunk_section"] = sec_chunk

    # ── RAGAS — Chat ──────────────────────────────────────────────────────
    if args.mode in ("ragas", "full"):
        ragas_chat = evaluate_ragas_chat(tool, CHAT_EVAL_DATASET)
        all_results["ragas_chat"] = ragas_chat

    # ── RAGAS — Section ───────────────────────────────────────────────────
    if args.mode in ("ragas_section", "full"):
        ragas_section = evaluate_ragas_section(tool, SECTION_EVAL_DATASET)
        all_results["ragas_section"] = ragas_section

    # ── Save results ──────────────────────────────────────────────────────
    save_results(all_results, args.mode)

    print(color("\n=== Evaluation complete ===\n", "bold"))


if __name__ == "__main__":
    main()