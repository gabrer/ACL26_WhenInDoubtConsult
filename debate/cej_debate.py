"""
CEJ Stage 2: Structured Debate 
Personas are exposed to all initial opinions and engage in critical
evaluation, revising their stance if persuaded.

Input:  Output of stage 1 (list of {"id", "text", "initial_opinions"}).
Output: JSON list of {"id", "text", "initial_opinions", "debate"}.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from cej_utils import (
    DEFINITION, OBJECTIVE,
    load_pipeline_json, load_examples, save_results, build_llm,
    format_opinions,
)


PROMPT_TEMPLATE = """You are continuing the expert panel discussion on the following text:

Text: "{text}"

Initial Opinions:
{initial_opinions}

Now, each persona must:
1. Read all other personas' initial opinions.
2. Reflect on whether their own reasoning is still the strongest.
3. Engage with at least one other persona by agreeing or disagreeing with their argument.
4. Update their stance if persuaded, or affirm their original decision.
5. Reassess and adjust their confidence accordingly.

Sexism Definition:
{definition}

Objective:
{objective}

{examples_section}

Important Notes:
- Confidence can be increased if supported by solid reasoning, or reduced if uncertainty arises.
- Final answers must state if the stance is changed or unchanged.

Output Example (per persona):
{{
  "persona": "Sexism Victim",
  "intent": "The author's intent is to shame the woman by dismissing her distress.",
  "reaction": "Agree with Linguist because their interpretation highlights the use of gendered stereotypes.",
  "updated_reasoning": "While my initial view focused on tone, I now realize the text uses the 'victim card' trope to discredit women's emotional responses.",
  "final_stance": "1 (changed from 0)",
  "updated_confidence": 0.72
}}

Respond ONLY with a JSON array of six objects in panel order. No other text."""


def parse_args():
    p = argparse.ArgumentParser(description="CEJ Stage 2: Structured persona debate")
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
        initial_opinions=format_opinions(entry["initial_opinions"]),
        definition=DEFINITION,
        objective=OBJECTIVE,
        examples_section=examples_section,
    )
    raw = llm.invoke(prompt).strip()
    return {
        "id": entry["id"],
        "text": entry["text"],
        "initial_opinions": entry["initial_opinions"],
        "debate": raw,
    }


def main():
    args = parse_args()
    llm = build_llm(args.model, args.temperature, args.ollama_url)
    data = load_pipeline_json(args.input_path)
    examples_section = load_examples(args.examples_path)
    print(f"Loaded {len(data)} instances, model={args.model}")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_one, entry, llm, examples_section): entry
            for entry in data
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Debate"):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[ERROR] id={futures[fut]['id']}: {e}")

    save_results(results, args.output_path)


if __name__ == "__main__":
    main()
