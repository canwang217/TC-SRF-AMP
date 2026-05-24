# Strict Structured Results

Canonical checkpoint:

`prot_flow/checkpoints/tc_srf_amp_strict_frozen_text_10k/tc_srf_amp_finetune_last_.pth`

Large structured evaluation at guidance scale 4 used 1024 samples per prompt:

- active_short_highpos: length ~9.98, net charge ~+5.07, KR fraction ~0.54, Cys ~0.06
- active_long_negative: length ~37.98, net charge ~-8.14, KR fraction ~0.04, Cys ~0.03
- active_medium_hydrophobic: length ~20.40, net charge ~+4.85, hydrophobic fraction ~0.47
- inactive_long_neutral: length ~37.00, net charge ~-3.91, KR fraction ~0.05

Guidance scale controls strength:

- low guidance gives milder control
- guidance 4 is the current recommended default
- guidance 6-8 can over-amplify charge, especially negative charge for long/low-KR prompts

Natural-language parser evaluation:

- `short_highpos` and `long_negative` are strongly controlled
- `medium_hydrophobic` is directionally controlled but high hydrophobicity match is weaker
- `inactive_long_neutral` is long/low-KR/no-Cys, but high guidance pushes charge negative

Recommended product path:

natural-language prompt -> parser -> structured prompt -> generate many candidates -> compute property metrics -> return filtered candidates.
