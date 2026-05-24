# Results

This directory contains compact summaries from the reported strict text-conditioned evaluations.

- `strict_frozen_text_10k_large/text_condition_diagnostics.json`: structured-prompt diagnostics and generation summary for 1024 samples per prompt.
- `strict_frozen_text_10k_large/physchem_summary_by_prompt.csv`: sequence-derived physicochemical summaries for the structured-prompt evaluation.
- `strict_nl_parser_g4_fixed/parsed_conditions.json`: natural-language prompts and their parsed structured prompts.
- `strict_nl_parser_g4_fixed/summary.json`: natural-language parser generation summaries for 512 samples per prompt.
- `strict_nl_parser_g4_fixed/physchem_summary_by_prompt.csv`: sequence-derived physicochemical summaries for the natural-language parser evaluation.

Full generated sequences are reproducible with the scripts under `prot_flow/scripts/` and are not committed to keep the repository lightweight.
