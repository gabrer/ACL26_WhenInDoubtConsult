# Collaborative Expert Judgment (CEJ) Pipeline

This directory implements the four-stage CEJ module.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The pipeline uses [Ollama](https://ollama.com/) for local LLM inference. Install Ollama and pull the models you plan to use:

```bash
ollama pull llama3.3:70b
ollama pull qwen2.5:72b
ollama pull cogito:70b
```

## Usage

The scripts accept both EXIST (JSON) and EDOS (CSV/TSV) data formats. Each stage reads the output of the previous stage.

### Full pipeline example

```bash
# Stage 1: Initial persona opinions
python cej_initial_opinions.py \
  --input_path data/routed_samples.json \
  --output_path outputs/initial_opinions.json \
  --model llama3.3:70b \
  --examples_file prompts/examples.txt

# Stage 2: Structured debate
python cej_debate.py \
  --input_path outputs/initial_opinions.json \
  --output_path outputs/debate.json \
  --model llama3.3:70b \
  --examples_file prompts/examples.txt

# Stage 3: Summarization
python cej_summarize.py \
  --input_path outputs/debate.json \
  --output_path outputs/summaries.json \
  --model llama3.3:70b

# Stage 4: Final judgment
python cej_judge.py \
  --input_path outputs/summaries.json \
  --output_path outputs/judgments.json \
  --model cogito:70b \
  --examples_file prompts/examples.txt
```

### Configurations 

**C3**: Qwen for personas + Cogito judge

```bash
python cej_initial_opinions.py --model qwen2.5:72b ...
python cej_debate.py --model qwen2.5:72b ...
python cej_summarize.py --model qwen2.5:72b ...
python cej_judge.py --model cogito:70b ...
```

**C4**: LLaMA for personas + Cogito judge

```bash
python cej_initial_opinions.py --model llama3.3:70b ...
python cej_debate.py --model llama3.3:70b ...
python cej_summarize.py --model llama3.3:70b ...
python cej_judge.py --model cogito:70b ...
```

## Examples File

The `--examples_file` argument accepts a plain text file with labeled examples to include in the classification prompt. This is optional; if omitted, the prompt relies on the sexism definition and guidelines alone.

Example format:

```text
Examples:

"Women belong in the kitchen."
Label: YES
Justification: Reinforces gendered domestic stereotypes.

"She won the Nobel Prize for her research."
Label: NO
Justification: Celebrates achievement without sexist framing.
```

## Input/Output Formats

**Stage 1 input**: JSON list/dict or CSV with `id` and `text` 

**Stage 1 output**:
```json
[{"id": "123", "text": "...", "initial_opinions": "<LLM output>"}]
```

**Stage 2 output**:
```json
[{"id": "123", "text": "...", "initial_opinions": "...", "debate": "<LLM output>"}]
```

**Stage 3 output**:
```json
[{"id": "123", "text": "...", "summary": "<LLM output>"}]
```

**Stage 4 output**:
```json
[{"id": "123", "text": "...", "label": 1, "justification": "...", "confidence": 0.92}]
```

## Personas

Six personas are used:

| Persona | Perspective |
|---|---|
| Normal Person | General public, everyday cultural norms |
| Linguist | Semantics, pragmatics, gendered language |
| Gender Studies Expert | Patriarchy, privilege, intersectionality |
| Legal Studies Expert | Anti-discrimination law, protected classes |
| Sexism Victim | Lived experience, emotional impact |
| Psychologist | Cognitive biases, psychological effects |
