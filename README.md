# AI in Cyber Security: Capstone Project (Unit 1)
**Repository:** `ai-cybersecurity-unit1-mini-project-anant-goel`  
**Dataset:** UNSW-NB15 Network Intrusion Detection Dataset  

---

## 📌 Project Overview
This project delivers an end-to-end AI-driven threat detection pipeline addressing real-world network cybersecurity challenges. Utilizing the benchmark **UNSW-NB15** intrusion dataset, this unit project investigates threat landscape characteristics, executes exploratory data analysis (EDA), implements supervised Machine Learning threat classification models, and benchmarks prototype Deep Learning neural architectures.

---

## 🏗️ Repository Architecture
```text
ai-cybersecurity-unit1-mini-project-anant-goel/
├── .gitignore                      # Git exclusion rules for venv, cache, & artifacts
├── requirements.txt                # Python environment package dependencies
├── README.md                       # Complete project and pipeline documentation
├── data/                           # Network intrusion datasets
│   ├── UNSW_NB15_training-set.csv  # 82,332 records with 45 network traffic attributes
│   └── UNSW_NB15_testing-set.csv   # 175,341 records for out-of-sample evaluation
├── src/                            # Scripts and analysis notebooks
│   ├── validate_env.py             # Environment verification script
│   └── ai_cybersecurity_analysis.ipynb # Interactive end-to-end Jupyter Notebook
```

---

## 🚀 Tasks Breakdown

### **Task 1: Workspace Bootstrap & Environment Setup**
- Created modular workspace structure: `data/`, `src/`, `.gitignore`, `requirements.txt`.
- Set up a Python 3.13 virtual environment (`.venv`).
- Installed core data science & security stack: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`.
- Validated setup using `src/validate_env.py`.

### **Task 2: Cyber Security Threat Landscape Analysis (EDA)**
- Loaded the UNSW-NB15 dataset (training & testing partitions).
- Inspected traffic flow features: duration (`dur`), packet counts (`spkts`/`dpkts`), byte volumes (`sbytes`/`dbytes`), packet rate, time-to-live (`sttl`/`dttl`), jitter, and TCP window metrics.
- Analyzed normal vs. suspicious traffic distributions across 9 attack categories:
  - *Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms*.
- Visualized class balance, attack distribution, correlation matrices, and distribution of attack-related metrics.

### **Task 3: Basic Machine Learning-Based Threat Classification**
- Preprocessing Pipeline:
  - Missing value imputation & sanitization.
  - Frequency / One-Hot Encoding for categorical features (`proto`, `service`, `state`).
  - Robust Feature Scaling (`StandardScaler` / `MinMaxScaler`) across high-dynamic-range network features.
- Supervised ML Models:
  - **Logistic Regression**: Linear baseline with regularized decision boundary.
  - **Decision Tree Classifier**: Non-linear, interpretable tree-based threat detector.
- Model Evaluation Metrics:
  - Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.

### **Task 4: Introduction to Deep Learning Concepts (Prototype Implementation)**
- Neural Network Architecture:
  - **Multi-Layer Perceptron (MLP)** via `scikit-learn.neural_network.MLPClassifier`.
  - Architecture: Dense input layer -> Hidden Layer 1 (ReLU, 64 neurons) -> Hidden Layer 2 (ReLU, 32 neurons) -> Output (Sigmoid / Softmax).
- Comparative Benchmark (ML vs. DL):
  - Training latency / inference speed.
  - Detection accuracy, false positive rate (FPR), and recall on zero-day attack classes.
  - Practical deployment feasibility in Security Operations Centers (SOC).

---

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Anant-Goel2006/ai-cybersecurity-unit1-mini-project-anant-goel.git
cd ai-cybersecurity-unit1-mini-project-anant-goel
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Validate Environment
```bash
python src/validate_env.py
```

### 5. Launch the Analysis Notebook
```bash
jupyter notebook src/ai_cybersecurity_analysis.ipynb
```
