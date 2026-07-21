# Vers une chaîne ECG multimodale déployable sur edge (fondations SMHFL)
## Volet Edge — Taher KHALLAF

### Description
The project is about deploying a CNN-1D + BiLSTM ECG classification model on edge (Raspberry Pi). The dataset is Chapman-Shaoxing 12-lead ECG (WFDB format). Python 3.14.

### Folder Structure
```
.
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── scripts/
│   └── extract_dataset.py
└── src/
    └── __init__.py
```

### Setup Instructions
1. Create venv: `python -m venv venv`
2. Activate venv: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
3. Install dependencies: `pip install -r requirements.txt`
4. Extract dataset: `python scripts/extract_dataset.py`
5. Train model (TODO)
6. Export ONNX (TODO)
7. Benchmark (TODO)

### Week-by-week Planning
- Week 1: Setup and Dataset Extraction
- Week 2: Model Training
- Week 3: ONNX Export
- Week 4: Benchmarking on Edge (Raspberry Pi)
- Week 5: MQTT Integration

### Key Commands
- `python scripts/extract_dataset.py`

### Links
- [Chapman-Shaoxing dataset on PhysioNet](https://physionet.org/content/ecg-arrhythmia/1.0.0/)

### License
MIT License (Placeholder)
