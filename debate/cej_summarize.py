"""
CEJ Stage 3: Debate Summarization
A dedicated summarization agent condenses the full debate into a concise
synthesis highlighting main arguments, points of consensus, and unresolved
disagreements.

Input:  Output of stage 2 (list of {"id", "text", "initial_opinions", "debate"}).
Output: JSON list of {"id", "text", "summary"}.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from cej_utils import (
    load_pipeline_json, save_results, build_llm, format_opinions,
)


PROMPT_TEMPLATE = """You are an expert summarizer tasked with condensing an expert panel discussion on sexism classification into a concise synthesis.

Text under discussion: "{text}"

=== INITIAL OPINIONS ===
{initial_opinions}

=== STRUCTURED DEBATE ===
{debate}

Your summary must include:
1. Each persona's initial opinion (label and confidence).
2. Each persona's final opinion after debate (label and confidence).
3. Whether their stance changed or remained the same, and if changed, who influenced them and why.
4. Whether there was consensus or disagreement among the panel.
5. The key arguments from each side.
6. Any major unresolved disagreements and their reasons.

Keep the summary structured and concise."""


def parse_args():
    p = argparse.ArgumentParser(description="CEJ Stage 3: Debate summarization")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--ollama_url", type=str, default=None)
    return p.parse_args()


def process_one(entry, llm):
    prompt = PROMPT_TEMPLATE.format(
        text=entry["text"],
        initial_opinions=format_opinions(entry.get("initial_opinions", "")),
        debate=entry.get("debate", ""),
    )
    summary = llm.invoke(prompt).strip()
    return {"id": entry["id"], "text": entry["text"], "summary": summary}


def main():
    args = parse_args()
    llm = build_llm(args.model, args.temperature, args.ollama_url)
    data = load_pipeline_json(args.input_path)
    print(f"Loaded {len(data)} instances, model={args.model}")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_one, entry, llm): entry
            for entry in data
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Summarizing"):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[ERROR] id={futures[fut]['id']}: {e}")

    save_results(results, args.output_path)


if __name__ == "__main__":
    main()
