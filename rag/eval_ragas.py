"""
RAGAS evaluation of the RAG pipeline.

Metrics evaluated:
  - Faithfulness       : is the answer grounded in the retrieved context?
  - Answer Relevancy   : does the answer address the question?
  - Context Precision  : are the retrieved chunks relevant to the ground truth?
  - Context Recall     : does the context cover the ground truth?

Usage:
    python rag/eval_ragas.py
"""


import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Test set — questions, ground truths drawn from the V-PCC / Gaussian splat
# documents in the corpus.
# ---------------------------------------------------------------------------
TEST_SET = [
    {
        "question": "What are the latest performances of V-PCC for Gaussian splatting?",
        "ground_truth": (
            "The latest performances of V-PCC for Gaussian splatting were evaluated "
            "using a cross-check conducted by Nokia on sequences including bartender, "
            "cinema, breakfast, and manwithfruit. The cross-check confirmed reproducibility "
            "and overall consistency of the reported results. A minor discrepancy was noted "
            "in the bitstream size of two Gaussian parameters in a single rate point of the "
            "bartender sequence. The work focuses on standardizing the V-PCC RAW toolset "
            "for Gaussian splatting through Amendment 1 to ISO/IEC 23090-5, defining new "
            "V3C attribute video components and reconstruction identifiers for Gaussian splats."
        ),
    },
    {
        "question": "Which sequences were used to evaluate V-PCC for Gaussian splatting?",
        "ground_truth": (
            "The sequences used to evaluate V-PCC for Gaussian splatting are bartender, "
            "cinema, breakfast, and manwithfruit."
        ),
    },
    {
        "question": "What does Amendment 1 to ISO/IEC 23090-5 propose for Gaussian splatting?",
        "ground_truth": (
            "Amendment 1 to ISO/IEC 23090-5 proposes to define coding for Gaussian splats "
            "within the V-PCC framework. It aims to define new V3C attribute video components "
            "specifically for Gaussian splats, new attributes and reconstruction identifiers, "
            "and guidance on their utilization during the reconstruction process. The goal is "
            "to standardize the V-PCC RAW toolset for Gaussian splatting without introducing "
            "new coding methods or V3C units."
        ),
    },
    {
        "question": "What is the V-PCC RAW toolset used for in the context of Gaussian splatting?",
        "ground_truth": (
            "The V-PCC RAW toolset is used to encode Gaussian splat sequences by placing "
            "spherical harmonics (SH) coefficients, rotation, and scale attributes into "
            "video frames. These are compressed using custom attribute SEI messages. The "
            "toolset adapts existing V-PCC RAW tools to the Gaussian splatting use case "
            "without introducing new coding methods or V3C units."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helper: run the full RAG pipeline for one question
# ---------------------------------------------------------------------------
def run_rag(question: str) -> tuple[str, list[str]]:
    """Return (answer_text, list_of_context_strings)."""
    from query import retrieve_from_all_indexes, get_llm, _RAG_PROMPT, format_docs, detect_language

    _, lang_hint = detect_language(question)
    docs = retrieve_from_all_indexes(question, verbose=False)
    context_text = format_docs(docs)
    contexts = [doc.page_content for doc in docs]

    llm, _ = get_llm()
    chain = _RAG_PROMPT | llm

    answer = ""
    for chunk in chain.stream({"question": question + lang_hint, "context": context_text}):
        if hasattr(chunk, "content"):
            answer += chunk.content
        elif hasattr(chunk, "text"):
            answer += chunk.text
        else:
            answer += str(chunk)

    return answer.strip(), contexts


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RAGAS Evaluation")
    print("=" * 60)

    # --- Step 1: collect RAG outputs for each test question ---
    print(f"\nRunning RAG pipeline for {len(TEST_SET)} test questions...")
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, sample in enumerate(TEST_SET, 1):
        q = sample["question"]
        gt = sample["ground_truth"]
        print(f"\n[{i}/{len(TEST_SET)}] {q}")
        answer, contexts = run_rag(q)
        print(f"  → answer length: {len(answer)} chars, {len(contexts)} context chunks")
        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(gt)

    # --- Step 2: build RAGAS dataset ---
    from datasets import Dataset
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })

    # --- Step 3: configure RAGAS with our local LLM + embeddings ---
    print("\nConfiguring RAGAS with local Qwen LLM and multilingual embeddings...")

    import warnings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    from query import get_llm, get_embeddings
    llm, model_name = get_llm()
    embeddings = get_embeddings("other")

    wrapped_llm = LangchainLLMWrapper(llm)
    wrapped_emb = LangchainEmbeddingsWrapper(embeddings)

    print(f"  LLM: {model_name}")
    print(f"  Embeddings: {os.getenv('EMBEDDING_MODEL_ML', 'multilingual-e5-large-instruct')}")

    # --- Step 4: run evaluation ---
    print("\nRunning RAGAS evaluation (this may take 10–30 minutes)...")
    from ragas import evaluate

    from ragas import RunConfig
    # max_workers=1: run jobs sequentially — parallel jobs compete for the same GPU and all timeout
    # timeout=600: 10 min per LLM call to accommodate the slow local model
    run_config = RunConfig(timeout=600, max_retries=2, max_workers=1)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=wrapped_llm,
        embeddings=wrapped_emb,
        run_config=run_config,
    )

    # --- Step 5: display results ---
    print("\n" + "=" * 60)
    print("RAGAS RESULTS")
    print("=" * 60)
    df = result.to_pandas()
    print(f"Columns returned by RAGAS: {df.columns.tolist()}")
    print(df.to_string(index=False))
    print("\n--- Averages ---")
    skip = {"user_input", "response", "retrieved_contexts", "reference",
            "question", "answer", "contexts", "ground_truth"}
    metric_cols = [c for c in df.columns if c not in skip]
    for col in metric_cols:
        try:
            print(f"  {col:30s}: {df[col].mean():.4f}")
        except Exception:
            print(f"  {col:30s}: (non-numeric)")
    print("=" * 60)

    # Save to CSV
    out_path = Path("ragas_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
