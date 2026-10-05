# AI-CyberSecurity-unit1-mini-project-ANANTGOEL

[![Python 3.13](https://img.shields.io/badge/Python-3.13.2-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-red.svg)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end artificial intelligence and machine learning threat detection pipeline built for the **AI in Cyber Security Capstone Project (Unit 1)**. Utilizing the contemporary **UNSW-NB15** network intrusion benchmark, this repository implements automated workspace bootstrapping, deep behavioral threat landscape profiling (EDA), MITRE ATT&CK taxonomy mapping, supervised machine learning threat classification, deep learning neural prototype evaluation, and practical Security Operations Center (SOC) feasibility benchmarking.

---

## 📑 Table of Contents
- [Executive Summary](#-executive-summary)
- [Repository Architecture](#-repository-architecture)
- [Dataset Provenance & Schema](#-dataset-provenance--schema)
- [Task 1: Workspace Bootstrap & Setup](#-task-1-workspace-bootstrap--setup)
- [Task 2: Cyber Security Threat Landscape Analysis (EDA)](#-task-2-cyber-security-threat-landscape-analysis-eda)
  - [MITRE ATT&CK Taxonomy Mapping](#mitre-attck-taxonomy-mapping)
  - [Key Behavioral Threat Signatures](#key-behavioral-threat-signatures)
- [Task 3: Supervised Machine Learning Threat Classification](#-task-3-supervised-machine-learning-threat-classification)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Per-Attack-Category Detection Recall Audit](#per-attack-category-detection-recall-audit)
  - [SOC Alert Fatigue & Threshold Tuning](#soc-alert-fatigue--threshold-tuning)
- [Task 4: Deep Learning Prototype & Comparative Benchmark](#-task-4-deep-learning-prototype--comparative-benchmark)
  - [Consolidated Model Performance Table](#consolidated-model-performance-table)
  - [Out-of-Sample Generalization Stress Test (175k Flows)](#out-of-sample-generalization-stress-test-175k-flows)
  - [SOC Operational Feasibility & Trade-Offs](#soc-operational-feasibility--trade-offs)
- [Quickstart Guide & Execution](#-quickstart-guide--execution)
- [GitHub Remote Sync](#-github-remote-sync)

---

## 🎯 Executive Summary
Modern enterprise networks generate millions of telemetry flow records per second. Traditional signature-based Intrusion Detection Systems (IDS) fail against zero-day exploits, protocol obfuscation, and subtle persistent threat campaigns. This project develops and benchmarks:
1. **Linear Supervised Machine Learning (Logistic Regression)**: Rapid baseline ($91.73\%$ accuracy, $4.11$s training).
2. **Non-Linear Rule-Based Machine Learning (Decision Tree Classifier)**: High-accuracy, transparent white-box classifier ($96.57\%$ accuracy, $96.86\%$ F1, $2.49$s training, $2.77\%$ False Positive Rate).
3. **Deep Learning Multi-Layer Perceptron (MLP Neural Network)**: Representation learning ($95.60\%$ accuracy, $0.992$ ROC-AUC, $23.09$s training).
4. **Generalization Stress Testing**: Out-of-sample evaluation on 175,341 unseen network sessions, capturing real-world cyber domain shift ($89.58\%$ retained test accuracy).

---

## 🏗️ Repository Architecture
```text
AI-CyberSecurity-unit1-mini-project-ANANTGOEL/
├── .gitignore                          # Ignores .venv, cache, checkpoints, and logs
├── README.md                           # Master project documentation
├── requirements.txt                    # Verified Python environment dependencies
├── data/                               # Network intrusion benchmark datasets
│   ├── UNSW_NB15_training-set.csv      # 82,332 records across 45 attributes (14.7 MB)
│   └── UNSW_NB15_testing-set.csv       # 175,341 records for out-of-sample stress testing (30.8 MB)
└── src/                                # Source code and analytical notebooks
    ├── validate_env.py                 # Automated environment & file verification script
    ├── build_notebook.py               # Headless generator and executor for the capstone notebook
    └── ai_cybersecurity_capstone.ipynb # Master interactive Jupyter Notebook with all outputs & plots
```

---

## 📊 Dataset Provenance & Schema
The **UNSW-NB15** dataset was synthesized by the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS) using an IXIA PerfectStorm traffic generator. It overcomes critical flaws of legacy datasets (like KDD99/NSL-KDD) by reflecting modern network traffic mixes and realistic low-footprint attack vectors.

### 45-Attribute Functional Categories:
- **Flow Identifiers & Basic**: `id`, `dur` (duration), `proto` (protocol), `service` (application service), `state` (connection state).
- **Payload & Content**: `sbytes` (source bytes), `dbytes` (dest bytes), `sttl` (source TTL), `dttl` (dest TTL), `sloss`/`dloss` (packet drops), `swin`/`dwin` (TCP window sizes), `smean`/`dmean` (mean packet size).
- **Temporal & Velocity**: `rate` (packets/sec), `sload`/`dload` (bits/sec), `sinpkt`/`dinpkt` (inter-packet arrival), `sjit`/`djit` (packet jitter).
- **TCP Latency Metrics**: `tcprtt` (round-trip time), `synack` (SYN-to-SYN-ACK latency), `ackdat` (SYN-ACK-to-ACK latency).
- **Connection Counts**: `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `ct_src_ltm`, `ct_srv_dst`.
- **Target Labels**: `attack_cat` (multi-class category name), `label` (binary: 0 = Normal, 1 = Attack).

---

## 📦 Task 1: Workspace Bootstrap & Setup
- **Isolated Virtual Environment**: Built on Python 3.13.2 (`.venv`).
- **Installed Packages**: `numpy==2.5.3`, `pandas==3.0.6`, `scikit-learn==1.9.1`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `jupyter==1.1.1`, `ipykernel==7.3.0`.
- **Automated Validation**: Running `python src/validate_env.py` performs automated environment checks on Python paths, dependency imports, directory structure, and dataset existence.

---

## 🔍 Task 2: Cyber Security Threat Landscape Analysis (EDA)

### Class Distribution (Training Partition):
- **Normal / Benign (0)**: 37,000 flows ($44.94\%$)
- **Suspicious / Attack (1)**: 45,332 flows ($55.06\%$)

### MITRE ATT&CK Taxonomy Mapping:
| Attack Category | Train Count (%) | Test Count (%) | MITRE ATT&CK Tactic | Technique ID & Name | Security Description |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Generic** | 18,871 (22.9%) | 40,000 (22.8%) | Impact / Defense Evasion | **T1499** / **T1027** | Collision attacks against ciphers / protocol standards. |
| **Exploits** | 11,132 (13.5%) | 33,393 (19.0%) | Initial Access / Execution | **T1190** Exploit Public App | Targeting known CVE vulnerabilities in Web, SMB, or OS kernels. |
| **Fuzzers** | 6,062 (7.4%) | 18,184 (10.4%) | Execution / Discovery | **T1203** Client Execution | Submitting massive malformed payloads to discover buffer overflows. |
| **DoS** | 4,089 (5.0%) | 12,264 (7.0%) | Impact | **T1498** Network DoS | Flooding target networks with volumetric packets to exhaust bandwidth. |
| **Reconnaissance**| 3,496 (4.2%) | 10,491 (6.0%) | Reconnaissance / Discovery | **T1595** Active Scanning | Probing open ports and active services using Nmap / SYN scans. |
| **Analysis** | 677 (0.8%) | 2,000 (1.1%) | Discovery | **T1087** / **T1082** Discovery | Intrusions inspecting web application parameters and SQL errors. |
| **Backdoor** | 583 (0.7%) | 1,746 (1.0%) | Persistence / C2 | **T1059** / **T1105** C2 | Stealthy persistent access channels bypassing authentication. |
| **Shellcode** | 378 (0.5%) | 1,133 (0.6%) | Execution | **T1055** Process Injection | Small executable assembly payloads injected to spawn root shells. |
| **Worms** | 44 (0.05%) | 130 (0.07%) | Lateral Movement | **T1570** Lateral Tool Transfer | Self-replicating payloads scanning subnets to spread internally. |

### Key Behavioral Threat Signatures:
1. **Source Time-to-Live (`sttl`)**: Normal operating systems initialize TTL to 64 (Linux) or 128 (Windows). Attacks exhibit extreme spikes at boundary value **254** from raw packet generators (Scapy, Metasploit).
2. **Packet Velocity (`rate`)**: Malicious DoS and Fuzzing bursts produce extreme velocity spikes ($> 100,000$ packets/sec).
3. **Payload Asymmetry**: Normal browsing is downstream-heavy ($dbytes > sbytes$); data exfiltration and SYN flooding invert this pattern ($sbytes \gg dbytes$).
4. **Service Vulnerability**: DNS, HTTP, and raw IP transport protocols register the highest proportion of malicious flows.

---

## 🤖 Task 3: Supervised Machine Learning Threat Classification

### Preprocessing Pipeline:
- **Sanitization**: Removed `id`, `attack_cat`, and helper columns.
- **Categorical Handling**: Grouped rare protocols outside top 10 into `'other'` to mitigate feature explosion; One-Hot Encoded `proto`, `service`, `state`.
- **Feature Normalization**: Standardized all continuous telemetry via `StandardScaler` ($z = \frac{x - \mu}{\sigma}$).
- **Stratified Split**: 75% train (61,749 flows), 25% validation (20,583 flows).

### Per-Attack-Category Detection Recall Audit:
Which specific cyber threats slip past the detector?
| Attack Family | Total in Validation | Detected | Detection Recall (%) | Missed Breaches | Vulnerability Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Worms** | 5 | 5 | **100.0%** | 0 | Distinct lateral scanning traffic pattern caught cleanly. |
| **Generic** | 4,678 | 4,666 | **99.74%** | 12 | Standard packet structures easily isolated by tree splits. |
| **Analysis** | 177 | 176 | **99.44%** | 1 | High detection on web scanning parameters. |
| **Reconnaissance** | 850 | 834 | **98.12%** | 16 | Rapid port probes detected via connection count features. |
| **DoS** | 1,006 | 984 | **97.81%** | 22 | Extreme packet velocity triggers clear threshold splits. |
| **Backdoor** | 161 | 157 | **97.52%** | 4 | Low volume persistent connections largely identified. |
| **Exploits** | 2,793 | 2,687 | **96.20%** | 106 | Moderate false negatives due to payload encryption/polymorphism. |
| **Shellcode** | 98 | 88 | **89.80%** | 10 | Compact memory-injection payloads occasionally blend with normal traffic. |
| **Fuzzers** | 1,565 | 1,287 | **82.24%** | 278 | **Lowest Recall**: Fuzzers intentionally randomize packet sizes and timings to evade signature filters. |

### Top 5 Predictive Features (Gini Importance):
1. `sttl` ($34.12\%$ of total tree decisions): Forged source TTL header values.
2. `ct_state_ttl` ($18.45\%$): Cross-state time-to-live consistency.
3. `dmean` ($11.20\%$): Mean destination packet payload size.
4. `sbytes` ($7.85\%$): Outbound byte volume.
5. `rate` ($5.30\%$): Transmission velocity in packets/sec.

### SOC Alert Fatigue & Threshold Tuning:
- **Cost Tradeoff**: $Cost(FN) \approx \$50,000$ (unmitigated breach) vs $Cost(FP) \approx \$50$ (15-min analyst triage).
- **Optimal Threshold**: Shifting decision threshold from default $0.50$ to **$0.35$** increases Threat Recall to $98.4\%$, eliminating $60\%$ of missed breaches with only a negligible bump in false alarms.

---

## 🧠 Task 4: Deep Learning Prototype & Comparative Benchmark

### Multi-Layer Perceptron (MLP) Architecture:
- **Input Layer**: 69 preprocessed continuous & encoded features.
- **Hidden Layer 1**: 64 neurons (ReLU activation).
- **Hidden Layer 2**: 32 neurons (ReLU activation).
- **Output Layer**: 1 neuron (Logistic Sigmoid, Binary Cross-Entropy).
- **Optimizer**: Adam ($\eta=0.001$), batch size 256, early stopping enabled.

### Consolidated Model Performance Table:
| Model Architecture | In-Sample Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Positive Rate | Training Time | Inference Latency (10k flows) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Linear)** | 91.73% | 92.68% | 92.27% | 92.47% | 0.9773 | 8.93% | ~4.11 s | **~5.31 ms** |
| **Decision Tree (Non-Linear)** | **96.57%** | **97.70%** | **96.04%** | **96.86%** | 0.9912 | **2.77%** | **~2.49 s** | ~16.31 ms |
| **MLP Neural Network (Deep Learning)** | 95.60% | 96.22% | 95.77% | 96.00% | **0.9921** | 4.61% | ~23.09 s | ~27.54 ms |

### Out-of-Sample Generalization Stress Test (175k Flows):
Evaluated against the official benchmark `UNSW_NB15_testing-set.csv` (175,341 unseen network sessions):
| Model Architecture | In-Sample Val Accuracy | Out-of-Sample Test Accuracy | Out-of-Sample Test F1 | Accuracy Shift (Domain Drop) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 91.73% | 87.53% | 90.12% | -4.20% |
| **Decision Tree Classifier** | **96.57%** | **89.58%** | **91.79%** | -7.00% |
| **MLP Neural Network** | 95.60% | 89.18% | 91.48% | -6.42% |

*Takeaway: Out-of-sample testing reveals the classic **Cyber Domain Shift**. Unseen network sessions with differing background load cause a ~6-7% accuracy drop, demonstrating why real-world SOC AI systems require continuous telemetry retraining.*

---

## 🔬 SOC Operational Feasibility & Trade-Offs

1. **Training Latency vs. Detection Accuracy**:
   - The **Decision Tree** is superior for fast iterative training ($2.49$ seconds) and achieves the lowest False Positive Rate ($2.77\%$).
   - The **MLP Neural Network** achieves the highest ROC-AUC ($0.9921$), showing superior probability calibration, but takes $\approx 9\times$ longer to train ($23.09$ seconds).
2. **Hardware Constraints (Inline Edge IPS vs Centralized Cloud SIEM)**:
   - **Inline Next-Gen Firewalls (Palo Alto / Fortinet)**: Require microsecond packet inspection budgets ($< 1$ ms). Decision trees compile directly into hardware **eBPF** or ASIC lookup tables without GPU coprocessors.
   - **Centralized Cloud SIEM (Splunk / Microsoft Sentinel)**: Deep Learning representations excel at correlating multi-stage Advanced Persistent Threats (APTs) across disparate log sources.
3. **White-Box Explainability**:
   - When an automated firewall blocks critical business traffic, analysts must justify the action. Decision Trees provide human-auditable logic:
     $$\text{IF } sttl > 250 \text{ AND } ct\_state\_ttl > 2 \implies \text{Block (Exploit)}$$
     Neural networks provide only a black-box probability score, increasing analyst verification overhead.
4. **Recommended Hybrid Two-Tier Enterprise Architecture**:
   - **Tier 1 (Perimeter Inline IPS)**: Deploy high-speed Decision Trees to drop 98% of obvious volumetric threats (DoS, Scans, Generic exploits) at wire-speed with sub-millisecond latency.
   - **Tier 2 (Core Behavioral SIEM Engine)**: Route ambiguous, low-conviction sessions to Deep Learning models for zero-day behavioral analysis and multi-vector correlation.

---

## 🚀 Quickstart Guide & Execution

### 1. Prerequisites
Ensure Python 3.10+ and Git are installed on your machine.

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Anant-Goel2006/AI-CyberSecurity-unit1-mini-project-ANANTGOEL.git
cd AI-CyberSecurity-unit1-mini-project-ANANTGOEL

# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\activate

# Activate environment (Linux / macOS)
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Validate Environment
```bash
python src/validate_env.py
```

### 5. Launch the Interactive Jupyter Notebook
```bash
jupyter notebook src/ai_cybersecurity_capstone.ipynb
```

---

## 🌐 GitHub Remote Sync

To push this repository to your GitHub account:

```bash
git remote add origin https://github.com/Anant-Goel2006/AI-CyberSecurity-unit1-mini-project-ANANTGOEL.git
git branch -M main
git push -u origin main
```
