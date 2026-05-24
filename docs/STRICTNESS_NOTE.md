# Strictness Note

The reported strict model follows the PDF-style conditioning route:

```text
structured text prompt -> SciBERT -> T_c + c_g -> cross-attention/AdaLN flow matching -> decoder
```

No direct label ids or direct attribute ids are supplied in strict training, diagnosis, or natural-language evaluation.

The codebase contains checkpoint-compatibility parameters from later exploratory direct-attribute experiments, such as `length_embedding` and `charge_embedding`. They are retained because the final checkpoint was saved with that model class and its EMA state expects those parameters. In strict scripts, these modules are inactive because:

```bash
TC_SRF_USE_LABEL_CONDITION=0
TC_SRF_USE_ATTRIBUTE_CONDITION=0
```

and diagnostic reports confirm:

```json
"attribute_ids_supplied": false,
"label_ids_supplied": false
```

Therefore, the strict results should be interpreted as PDF-style `T_c/c_g` text conditioning with a structured text prompt frontend, not as direct attribute embedding conditioning.
