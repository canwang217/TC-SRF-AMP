# Reproduce From Zero

This project uses pretrained ESM-2 and SciBERT as foundation encoders. “From zero” here means rebuilding the AMP generation pipeline from the provided raw peptide/AMP data and local pretrained model weights. It does not mean pretraining ESM-2 or SciBERT themselves.

## Level 1: Reproduce Final Results From Included Checkpoints

This is the fastest verification path.

```bash
python prot_flow/scripts/download_models.py
cd prot_flow
python -m py_compile \
  scripts/diagnose_text_conditioning_path.py \
  scripts/nl_condition_parser.py \
  scripts/run_strict_nl_prompt_benchmark.py \
  scripts/analyze_generated_physchem.py

bash scripts/run_eval_strict_structured.sh
bash scripts/run_eval_strict_nl_parser.sh
```

Canonical checkpoint:

```text
prot_flow/checkpoints/tc_srf_amp_strict_frozen_text_10k/tc_srf_amp_finetune_last_.pth
```

## Level 2: Rebuild Structured AMP Training Data

The training data for the strict text model is structured text paired with AMP sequences.

Full JSONL training files are large and are restored by `python prot_flow/scripts/download_models.py`.

```bash
cd prot_flow
python scripts/prepare_structured_amp_data.py \
  --train-input ../ensemble_train.jsonl \
  --valid-input ../ensemble_test.jsonl \
  --train-output ../ensemble_train_structured.jsonl \
  --valid-output ../ensemble_test_structured.jsonl \
  --summary-json ../structured_amp_summary.json
```

The structured text format is:

```text
activity: active; target: unknown; length_bin: short; charge: high_positive; kr_ratio: high; hydrophobicity: medium; cys: none; toxicity: unknown
```

This is still ordinary text input to SciBERT. It is not a direct attribute embedding in the strict path.

## Level 3: Re-train Strict PDF + Structured Prompt Model

This starts from the provided unconditional flow checkpoint and re-trains the text-conditioned AMP model.

```bash
cd prot_flow
bash scripts/run_train_strict_frozen_text_10k.sh
```

Critical settings:

```bash
TC_SRF_FREEZE_TEXT_ENCODER=1
TC_SRF_USE_LABEL_CONDITION=0
TC_SRF_USE_ATTRIBUTE_CONDITION=0
TC_SRF_AMP_TRAIN_JSONL=../ensemble_train_structured.jsonl
TC_SRF_AMP_VALID_JSONL=../ensemble_test_structured.jsonl
```

The text encoder is frozen because the unfrozen text checkpoint collapsed: different prompts produced nearly identical BERT embeddings. Freezing SciBERT preserves text differences and lets the flow model learn how to use `T_c`.

## Level 4: Re-train Base Peptide Components

The code for the base peptide latent pipeline is included.

Train compressor:

```bash
cd prot_flow
python train_compressor.py
```

Train decoder:

```bash
cd prot_flow
python train_decoder.py
```

Train unconditional flow:

```bash
cd prot_flow
bash scripts/run_pretrain_flow_from_existing_assets.sh
```

Then fine-tune the strict text model:

```bash
export TC_SRF_PRETRAIN_CKPT="$PWD/checkpoints/flow_matching/flow_peptides_pretrain_last_.pth"
bash scripts/run_train_strict_frozen_text_10k.sh
```

## Level 5: Natural Language Interface

Natural-language prompts are first parsed into structured text prompts.

Single prompt generation:

```bash
cd prot_flow
python scripts/generate_strict_from_prompt.py \
  --prompt "Generate a short highly cationic antimicrobial peptide rich in lysine and arginine with no cysteine." \
  --num 32 \
  --seed 0 \
  --guidance-scale 4
```

Benchmark reproduction:

```bash
cd prot_flow
bash scripts/run_eval_strict_nl_parser.sh
```

The parser output is stored in:

```text
generated_seqs/diagnostics/strict_nl_parser_g4_fixed/parsed_conditions.json
```

Generation and summaries are stored in:

```text
generated_seqs/diagnostics/strict_nl_parser_g4_fixed/combined.json
generated_seqs/diagnostics/strict_nl_parser_g4_fixed/summary.json
```

## Level 6: Physicochemical Analysis

```bash
cd prot_flow
python scripts/analyze_generated_physchem.py \
  --input generated_seqs/diagnostics/strict_nl_parser_g4_fixed/combined.json \
  --output-dir generated_seqs/diagnostics/strict_nl_parser_g4_fixed/physchem \
  --group-key prompt_name
```

This computes direct sequence-derived properties such as pI, pH7 charge, GRAVY, hydrophobic moment, acidic fraction, aromatic fraction, and entropy.
