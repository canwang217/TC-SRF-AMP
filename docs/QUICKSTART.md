# Quickstart

This is the shortest path for a user who only wants to generate AMP sequences from text.

## 1. Enter The Project

```bash
git clone https://github.com/canwang217/TC-SRF-AMP.git
cd TC-SRF-AMP/prot_flow
conda env create -f environment.yaml
conda activate tc-srf-amp
```

If the environment already exists on the server, just activate it.

Download the model assets:

```bash
python scripts/download_models.py
```

From the repository root, the equivalent command is:

```bash
python prot_flow/scripts/download_models.py
```

## 2. Generate From One Natural-Language Prompt

```bash
python scripts/generate_strict_from_prompt.py \
  --prompt "Generate a short highly cationic antimicrobial peptide rich in lysine and arginine with no cysteine." \
  --num 32 \
  --seed 0 \
  --guidance-scale 4
```

The script prints:

- the parsed structured prompt
- summary statistics for the generated sequences
- example generated peptide sequences
- the JSON output path

Default checkpoint:

```text
checkpoints/tc_srf_amp_strict_frozen_text_10k/tc_srf_amp_finetune_last_.pth
```

Default output:

```text
generated_seqs/user_prompts/user_prompt_g4_seed0.json
```

## 3. Check Only The Parser

```bash
python scripts/nl_condition_parser.py \
  --prompt "Generate a long negatively charged peptide with low lysine and arginine content and no cysteine."
```

This shows how free text is normalized into:

```text
activity; target; length_bin; charge; kr_ratio; hydrophobicity; cys; toxicity
```

The parser is rule-based. It does not require training.

## 4. Reproduce The Reported Structured Results

```bash
bash scripts/run_eval_strict_structured.sh
```

This uses fixed structured prompts and writes:

```text
generated_seqs/diagnostics/strict_frozen_text_10k_large/
```

## 5. Reproduce The Natural-Language Parser Results

```bash
bash scripts/run_eval_strict_nl_parser.sh
```

This runs:

```text
natural language -> parser -> structured prompt -> SciBERT -> T_c/c_g -> flow -> decoder
```

and writes:

```text
generated_seqs/diagnostics/strict_nl_parser_g4_fixed/
```

## 6. Compute Extra Physicochemical Properties

```bash
python scripts/analyze_generated_physchem.py \
  --input generated_seqs/diagnostics/strict_nl_parser_g4_fixed/combined.json \
  --output-dir generated_seqs/diagnostics/strict_nl_parser_g4_fixed/physchem \
  --group-key prompt_name
```

This computes direct sequence-derived properties including pH7 charge, pI, GRAVY, hydrophobic moment, acidic fraction, aromatic fraction, and entropy.

## Strict Path Guarantee

The convenience generation script uses the same strict path as the evaluation scripts:

```text
natural language -> structured text -> SciBERT -> token condition T_c + global condition c_g
```

It records parsed attribute IDs in the output JSON for readability, but it does not pass those IDs into the model. The model receives only text-derived `T_c` and `c_g`.
