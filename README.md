# AI-CyberSecurity-unit1-mini-project-ANANTGOEL

[![Python 3.13](https://img.shields.io/badge/Python-3.13.2-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-red.svg)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end AI and machine learning threat detection pipeline for the **AI in Cyber Security Capstone Project (Unit 1)**. Using the **UNSW-NB15** network intrusion benchmark, this project implements deep behavioral threat profiling (EDA), MITRE ATT&CK taxonomy mapping, supervised ML threat classification, deep learning neural prototype evaluation, and SOC feasibility benchmarking.

---

## 📁 Project Structure

```
AI-CyberSecurity-unit1-mini-project-ANANTGOEL/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── UNSW_NB15_training-set.csv      # 82,332 records × 45 features (14.7 MB)
│   └── UNSW_NB15_testing-set.csv       # 175,341 records × 45 features (30.8 MB)
└── notebooks/
    ├── 01_data_analysis_and_eda.ipynb           # EDA & Threat Landscape Profiling
    ├── 02_machine_learning_models.ipynb          # ML Classification & Interpretability
    └── 03_deep_learning_and_soc_benchmarks.ipynb # Deep Learning & SOC Feasibility
```

---

## 📓 Notebooks Overview

### 1. Data Analysis & EDA (`01_data_analysis_and_eda.ipynb`)
- Environment setup and 45-feature schema ontology audit
- Class distribution analysis (Normal vs Attack in train/test)
- 9 cyber attack families mapped to **MITRE ATT&CK** tactics and techniques
- Multi-feature behavioral signature profiling per threat category
- 4-panel threat indicator visualizations (TTL anomalies, packet velocity, byte asymmetry, service vulnerabilities)
- Pearson correlation heatmap and top-10 predictive features

### 2. Machine Learning Models (`02_machine_learning_models.ipynb`)
- Enterprise preprocessing pipeline: `StandardScaler` + `OneHotEncoder` via `ColumnTransformer`
- High-cardinality protocol grouping (131 → 11 categories)
- Stratified 75/25 train-validation split (61,749 / 20,583 flows)
- **Logistic Regression** (91.73% accuracy) and **Decision Tree** (96.57% accuracy) training
- Confusion matrices, ROC-AUC, PR-AUC evaluation
- Top-15 Gini feature importances with white-box rule interpretability
- Per-attack-category False Negative audit across all 9 threat families
- SOC decision threshold optimization (Cost(FN) >> Cost(FP) analysis)

### 3. Deep Learning & SOC Benchmarks (`03_deep_learning_and_soc_benchmarks.ipynb`)
- **MLP Neural Network** (64 → 32 ReLU neurons, Adam optimizer, early stopping)
- Loss convergence curve analysis
- Tri-model comparative benchmark table (Accuracy, Precision, Recall, F1, ROC-AUC, FPR, training time, inference latency)
- Comparative ROC and Precision-Recall curves
- Out-of-sample generalization stress test on 175,341 unseen network sessions
- SOC operational feasibility synthesis:
  - Inline edge IPS (eBPF/ASIC) vs centralized SIEM trade-offs
  - White-box explainability vs black-box alert fatigue
  - Adversarial robustness analysis
  - **Recommended Two-Tier Hybrid Enterprise Defense Architecture**

---

## 📊 Key Results

| Model | Accuracy | Recall | F1-Score | ROC-AUC | FPR | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 91.73% | 92.27% | 92.47% | 0.9773 | 8.93% | ~4s |
| **Decision Tree** | **96.57%** | **96.04%** | **96.86%** | 0.9912 | **2.77%** | **~2.5s** |
| MLP Neural Network | 95.60% | 95.77% | 96.00% | **0.9921** | 4.61% | ~23s |

---

## 📊 Dataset: UNSW-NB15

The **UNSW-NB15** dataset was created by the Australian Centre for Cyber Security (ACCS) using an IXIA PerfectStorm traffic generator. It captures 9 modern attack families: Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoors, Shellcode, and Worms.

---

## 🚀 Quickstart

```bash
# Clone
git clone https://github.com/Anant-Goel2006/AI-CyberSecurity-unit1-mini-project-ANANTGOEL.git
cd AI-CyberSecurity-unit1-mini-project-ANANTGOEL

# Setup virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\activate

# Activate (Linux / macOS)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter notebook notebooks/
```

---

## 🌐 GitHub Remote Sync

```bash
git remote add origin https://github.com/Anant-Goel2006/AI-CyberSecurity-unit1-mini-project-ANANTGOEL.git
git branch -M main
git push -u origin main
```
