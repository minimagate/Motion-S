# External real-world data

Kine does not redistribute third-party datasets. The transfer experiment uses the official ETH
Walking Pedestrians (EWAP) light annotation archive:

- Source: `https://ethz.ch/content/dam/ethz/special-interest/itet/cvl/vision-dam/documents/ewap_dataset_light.tgz`
- SHA-256 observed on 2026-10-25:
  `3e057d7c8631c9b5ae638b0d74d454d7be567532a51cd5e06f0323bfe863ce98`
- Expected extraction path: `data/real/ewap_dataset/`

Download and verify it with:

```bash
.venv/bin/kine-download-eth
```

The upstream README describes ground-plane positions in metres and annotation at 2.5 Hz. Dataset
terms and attribution remain governed by ETH Zurich's distribution.

