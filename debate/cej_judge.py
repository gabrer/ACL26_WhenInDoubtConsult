"""
CEJ Stage 4: Final Judgment 
The judge model synthesizes all available evidence (original text,
debate summary) to produce a final classification, justification,
and confidence score.

Input:  Output of stage 3 (list of {"id", "text", "summary"}).
Output: JSON list of {"id", "text", "label", "justification", "confidence"}.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from cej_utils import (
    DEFINITION, OBJECTIVE,
    load_pipeline_json, load_examples, save_results, build_llm,
    parse_json_response,
)


PROMPT_TEMPLATE = """You are an impartial judge of the detection of sexist language. Review the following expert panel summary and the original text being discussed then classify it for sexism.

You have the following information:

Text: "{text}"

Discussion summary:
{summary}

Your task:
1. Read the guidelines below carefully.
2. Review the expert panel summary.
3. Analyze the text for sexism.
4. Think before responding.
5. Decide the final label: 1 (sexist) or 0 (not sexist).
6. Provide a brief justification for your decision.
7. State your confidence score (0.0-1.0).

Sexism Definition:
{definition}

Objective:
{objective}

{examples_section}

Output: Provide only a valid JSON object like the following:
{{
  "label": 0,
  "justification": "Although the text contains profanity and strong language, it is not clearly directed at women. Based on the personas' disagreement and the classification guidelines, the text is offensive but not sexist.",
  "confidence": 0.79
}}"""


def parse_args():
    p = argparse.ArgumentParser(description="CEJ Stage 4: Final judge decision")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--examples_path", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--ollama_url", type=str, default=None)
    return p.parse_args()


def process_one(entry, llm, examples_section):
    prompt = PROMPT_TEMPLATE.format(
        text=entry["text"],
        summary=entry.get("summary", ""),
        definition=DEFINITION,
        objective=OBJECTIVE,
        examples_section=examples_section,
    )
    raw = llm.invoke(prompt).strip()
    parsed = parse_json_response(raw)
    if parsed and isinstance(parsed, dict):
        return {
            "id": entry["id"],
            "text": entry["text"],
            "label": parsed.get("label"),
            "justification": parsed.get("justification", ""),
            "confidence": parsed.get("confidence"),
        }
    return {
        "id": entry["id"],
        "text": entry["text"],
        "label": None,
        "justification": raw,
        "confidence": None,
    }


def main():
    args = parse_args()
    llm = build_llm(args.model, args.temperature, args.ollama_url)
    data = load_pipeline_json(args.input_path)
    examples_section = load_examples(args.examples_path)
    print(f"Loaded {len(data)} instances, judge model={args.model}")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_one, entry, llm, examples_section): entry
            for entry in data
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[ERROR] id={futures[fut]['id']}: {e}")

    save_results(results, args.output_path)


if __name__ == "__main__":
    main()
