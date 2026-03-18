"""
CEJ Stage 1: Initial Persona Opinions 
Each persona independently analyzes the input and provides a classification,
justification, and confidence score.

Input:  JSON/CSV with id + text fields (EXIST or EDOS format).
Output: JSON list of {"id", "text", "initial_opinions": {persona: raw_response}}.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from cej_utils import (
    PERSONAS, DEFINITION, OBJECTIVE,
    load_data, load_examples, save_results, build_llm,
)


PROMPT_TEMPLATE = """You are {persona_description} tasked with classifying the following text for sexism.

Text: "{text}"

Your task:
1. Read the guidelines below carefully.
2. Analyze the text carefully for sexism and the author's intent.
3. Think before responding.
4. Decide the final label: 1 (sexist) or 0 (not sexist).
5. Provide a short justification for your label based on your role.
6. Output a confidence score between 0.0 and 1.0 reflecting your certainty.

Sexism Definition:
{definition}

Objective:
{objective}

{examples_section}

Output: Provide only a valid JSON object like the following example:
{{
  "persona": "{persona_name}",
  "label": "1",
  "justification": "The text stereotypes women's intelligence.",
  "confidence": "0.87"
}}"""


def parse_args():
    p = argparse.ArgumentParser(description="CEJ Stage 1: Initial persona opinions")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--examples_path", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--ollama_url", type=str, default=None)
    return p.parse_args()


def process_one(row, llm, examples_section):
    opinions = {}
    for name, desc in PERSONAS.items():
        prompt = PROMPT_TEMPLATE.format(
            persona_description=desc,
            persona_name=name,
            text=row["text"],
            definition=DEFINITION,
            objective=OBJECTIVE,
            examples_section=examples_section,
        )
        opinions[name] = llm.invoke(prompt).strip()
    return {"id": row["id"], "text": row["text"], "initial_opinions": opinions}


def main():
    args = parse_args()
    llm = build_llm(args.model, args.temperature, args.ollama_url)
    rows = load_data(args.input_path)
    examples_section = load_examples(args.examples_path)
    print(f"Loaded {len(rows)} instances, model={args.model}")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_one, row, llm, examples_section): row
            for row in rows
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Initial opinions"):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[ERROR] id={futures[fut]['id']}: {e}")

    save_results(results, args.output_path)


if __name__ == "__main__":
    main()
