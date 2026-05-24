# Structured Prompt Module

The structured prompt module is an input normalization layer. It converts either sequence-derived training attributes or user natural language into a consistent text format.

It does not add a new neural condition branch in the strict PDF path.

## Training Data Construction

`scripts/prepare_structured_amp_data.py` reads records with:

```json
{"src": "...", "trg": "PEPTIDESEQUENCE"}
```

For each peptide sequence, it computes simple sequence attributes:

- length bin from sequence length
- charge bin from approximate net charge
- KR ratio bin from K/R fraction
- hydrophobicity bin from hydrophobic residue fraction
- cysteine bin from C count
- activity, target, toxicity from available source text when possible

It then renders one structured text prompt:

```text
activity: active; target: unknown; length_bin: short; charge: high_positive; kr_ratio: high; hydrophobicity: medium; cys: none; toxicity: unknown
```

The model is trained on:

```text
structured text prompt -> AMP sequence
```

## Natural Language Parsing

`scripts/nl_condition_parser.py` maps user text to the same fields.

Example:

```text
Generate a short highly cationic antimicrobial peptide rich in lysine and arginine with no cysteine.
```

becomes:

```text
activity: active; target: unknown; length_bin: short; charge: high_positive; kr_ratio: high; hydrophobicity: medium; cys: none; toxicity: unknown
```

The structured prompt is then passed to SciBERT exactly like any other text input.

The parser is rule-based. It does not require training. At generation time, parsed field IDs are only saved for inspection; they are not supplied to the flow model in the strict path.

## Relation To The PDF Architecture

The PDF architecture uses:

```text
Text prompt -> text encoder -> condition adapter -> T_c and c_g
```

Our structured prompt module only standardizes the text before this step:

```text
Natural language -> structured text prompt -> SciBERT -> T_c and c_g
```

The strict reported model does not pass `length_bin_id`, `charge_bin_id`, `kr_bin_id`, `hydrophobicity_bin_id`, `cys_bin_id`, or `label` into the flow model. The diagnostic scripts explicitly record:

```json
{
  "attribute_ids_supplied": false,
  "label_ids_supplied": false,
  "uses_text_encoder": true,
  "uses_token_condition_Tc": true,
  "uses_global_condition_cg": true
}
```

## Why This Is Useful

Free natural language is variable and ambiguous. Structured text preserves a natural-language interface while giving the text encoder a stable, repeated format seen during training. This made the strict `T_c/c_g` route controllable without adding direct attribute embeddings to the generation path.
