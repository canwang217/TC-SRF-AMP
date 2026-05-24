# Code Map

This release is organized around the PDF-style text-conditioned semantic rectified flow pipeline.

## Text Conditioning

- `prot_flow/model/text_conditioning.py`
  - `TextConditionEncoder`: SciBERT tokenizes and encodes the text prompt.
  - `token_projector`: converts token hidden states into token-level condition `T_c`.
  - `global_projector`: converts the `[CLS]` hidden state into global condition `c_g`.
  - `AdaptiveLayerNorm`: injects `c_g` into the flow Transformer.
  - `NullConditioning`: provides the unconditional condition for classifier-free guidance.

## Conditional Flow Matching

- `prot_flow/model/fm_estimator.py`
  - `TransformerEncoder`: peptide latent self-attention plus text cross-attention.
  - `BertBlock.crossattention`: lets peptide latent tokens attend to `T_c`.
  - Adaptive normalization layers use `c_g`.
  - The file also contains checkpoint-compatibility parameters for older direct attribute experiments. The strict scripts do not pass label or attribute ids, so these modules are inactive in the reported PDF-style results.

- `prot_flow/flow_matching_utils/flow_matching_holder.py`
  - `_prepare_conditioning`: builds `T_c`, `c_g`, and text attention masks.
  - `_apply_cfg_dropout`: classifier-free guidance dropout during training.
  - `calc_loss`: rectified flow matching loss between noise and peptide latent.
  - `pred_embeddings`: ODE-style sampling with classifier-free guidance.

## Peptide Semantic Space

- `prot_flow/encoders/`
  - ESM-2 encoding and sequence decoder wrappers.

- `prot_flow/compressors/`
  - Compressor/decompressor for the peptide latent space.

- `prot_flow/data/peptides_pretrain/`
  - Pretrain FASTA and normalization statistics used by the ESM latent pipeline.

## Training Entrypoints

- `prot_flow/train_compressor.py`: train peptide latent compressor.
- `prot_flow/train_decoder.py`: train sequence decoder.
- `prot_flow/train_flow_matching.py`: train unconditional peptide flow.
- `prot_flow/train_tc_srf_amp.py`: fine-tune text-conditioned AMP flow.
- `prot_flow/scripts/run_train_strict_frozen_text_10k.sh`: reproducible strict text fine-tune command.

## Structured Prompt And Natural Language

- `prot_flow/scripts/prepare_structured_amp_data.py`
  - Converts sequence-level attributes into structured text prompts for training.

- `prot_flow/scripts/nl_condition_parser.py`
  - Rule-based parser from natural language into structured text fields.

- `prot_flow/scripts/run_strict_nl_prompt_benchmark.py`
  - Natural language -> parser -> structured prompt -> strict `T_c/c_g` generator.

- `prot_flow/scripts/generate_strict_from_prompt.py`
  - Single-prompt user interface for generation. It calls the parser, sends only the structured text to the strict generator, and writes generated sequences plus summary metrics.

## Evaluation

- `prot_flow/scripts/diagnose_text_conditioning_path.py`
  - Checks text condition separation, vector-field sensitivity, and generated property summaries.

- `prot_flow/scripts/analyze_generated_physchem.py`
  - Computes direct physicochemical properties from generated sequences.

- `prot_flow/scripts/run_eval_strict_structured.sh`
  - Reproduces structured prompt results.

- `prot_flow/scripts/run_eval_strict_nl_parser.sh`
  - Reproduces natural-language parser results.
