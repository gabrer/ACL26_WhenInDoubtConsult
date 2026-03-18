"""
Shared constants and utilities for the Collaborative Expert Judgment (CEJ) pipeline.
Single source of truth for persona definitions, sexism definition,
classification objective, and data loading helpers.
"""

import json
import os
import re


PERSONAS = {
    "Normal Person": (
        "An average person (18-22 y/o male, Portugal, B.A.) with no "
        "specialized expertise in law, psychology, or linguistics."
    ),
    "Linguist": (
        "A linguist (23-45 y/o male, Poland, B.A. Linguistics) specializing "
        "in semantics, pragmatics, and discourse analysis, with a focus on "
        "gendered language."
    ),
    "Psychologist": (
        "A psychologist specializing in language, cognitive biases, and the "
        "psychological effects of sexism."
    ),
    "Legal Studies Expert": (
        "A legal expert (46+ y/o male, Portugal, M.A. Law) specializing in "
        "anti-discrimination laws, workplace regulations, and gender equality."
    ),
    "Gender Studies Expert": (
        "A gender studies expert (46+ y/o female, UK, B.A. Gender Studies) "
        "with deep knowledge of gender theories, power dynamics, and social "
        "structures."
    ),
    "Sexism Victim": (
        "A person (18-22 y/o female, South Africa, H.S. diploma) who has "
        "personally experienced sexism and understands its emotional and "
        "social impact."
    ),
}

DEFINITION = (
    'Sexism is defined as "prejudice, stereotyping or discrimination, '
    'typically against women, on the basis of sex." It can appear as:\n'
    '- Benevolent: "Women must be loved and respected, always treat them '
    'like a fragile glass."\n'
    '- Humorous: "You have to love women... just that... You will never '
    'understand them."\n'
    '- Hostile: "Humiliate, expose and degrade yourself as the fucking bitch '
    'you are if you want a real man to give you attention."\n\n'
    "Even the most subtle forms of sexism can be as harmful as violent ones. "
    "Not sexism: purely racist, homophobic, or non-gendered profanity."
)

OBJECTIVE = (
    "Classify whether the given text is sexist. Answer 1 (sexist) if the text "
    "is sexist itself, describes a sexist situation, or criticizes sexist "
    "behavior. Answer 0 (not sexist) if it is only racist, homophobic, or "
    "non-gendered hate, or uses profanity without targeting women or gender."
)


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        records = list(data.values())
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")

    rows = []
    for rec in records:
        if isinstance(rec, str):
            continue
        rid = str(rec.get("id") or rec.get("id_EXIST") or rec.get("rewire_id", ""))
        text = rec.get("text") or rec.get("tweet", "")
        if rid and text:
            rows.append({"id": rid, "text": str(text).strip()})
    return rows


def load_pipeline_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return list(data.values())


def load_examples(path):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f"Here are some examples:\n{f.read().strip()}"
    return ""


def save_results(results, path):
    results.sort(key=lambda x: str(x["id"]))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} records to {path}")


def parse_json_response(raw):
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"[\[{].*[\]}]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def format_opinions(opinions):
    if isinstance(opinions, dict):
        return "\n\n".join(f"{p}: {o}" for p, o in opinions.items())
    return str(opinions)


def build_llm(model, temperature, ollama_url=None, timeout=600):
    from langchain_ollama import OllamaLLM
    from langchain.globals import set_llm_cache
    from langchain_community.cache import InMemoryCache
    set_llm_cache(InMemoryCache())
    kwargs = dict(model=model, temperature=temperature,
                  keep_alive=timeout, timeout=timeout)
    if ollama_url:
        kwargs["base_url"] = ollama_url
    return OllamaLLM(**kwargs)
