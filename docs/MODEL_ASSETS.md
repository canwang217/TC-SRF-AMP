# Model And Data Assets

Large artifacts are hosted on Hugging Face:

```text
Canana040217/tc-srf-amp-models
```

The download script restores the expected layout:

```bash
python prot_flow/scripts/download_models.py
```

Required inference assets:

- `esm2_t12_35M_UR50D/`
- `scibert_scivocab_uncased/`
- `prot_flow/checkpoints/decoder-esm2-35M-peptides_pretrain.pth`
- `prot_flow/checkpoints/compressor/compressor-esm2-35M-peptides_pretrain.pth`
- `prot_flow/checkpoints/flow_matching/flow_peptides_pretrain_last_.pth`
- `prot_flow/checkpoints/tc_srf_amp_strict_frozen_text_10k/tc_srf_amp_finetune_last_.pth`

Required full retraining assets:

- `ensemble_train.jsonl`
- `ensemble_test.jsonl`
- `ensemble_train_structured.jsonl`
- `ensemble_test_structured.jsonl`
- `prot_flow/data/peptides_pretrain/train.fasta`

The GitHub repository only keeps small reference files and result summaries. Checkpoints, full JSONL data, full FASTA data, and pretrained encoder weights are intentionally excluded from Git.
