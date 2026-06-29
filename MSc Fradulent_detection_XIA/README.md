# An Explainable AI Framework for Fraud Detection and Transaction Risk Scoring in Digital Banking Platforms

## Project Overview
This project investigates the operational application of Explainable Artificial Intelligence
 (XAI) techniques within digital banking frameworks to detect fraud and compute real-time transaction risk scoring. 

Modern financial services are rapidly moving away from static, rule-based firewalls toward
 high-powered machine learning models. However, these complex ensemble models function as "black boxes," presenting 
significant regulatory liabilities. This study evaluates and builds an end-to-end 
pipeline that bridges the gap between predictive accuracy and interpretability. The ultimate objective is to deliver a 
framework that not only isolates fraudulent transactions with high precision but 
also generates transparent, compliant explanations to support internal risk audits, 
minimize investigator alert fatigue, and protect consumer trust.

---

## Research Aim
To design, develop, and evaluate an explainable machine learning framework that calculates granular transaction
 risk scores while satisfying global regulatory transparency mandates within digital banking infrastructures.

---

## Research Objectives
1. **Data Ingestion & Engineering:** Preprocess digital banking transaction streams, engineering critical 
behavioral and session-based features to isolate anomalies.
2. **Class Imbalance Mitigation:** Address the severe data skew characteristic of financial fraud
 (where legitimate transfers vastly outnumber fraudulent entries) using advanced oversampling and preprocessing techniques.
3. **Algorithmic Benchmarking:** Develop and evaluate multiple machine learning paradigms, ranging 
from inherently transparent models to complex tree-based ensembles.
4. **Performance Evaluation:** Assess classification capabilities using metrics optimized 
for highly imbalanced arrays, prioritizing minority-class capture.
5. **XAI Integration:** Layer post-hoc local and global explainability methodologies over 
the peak-performing engine to demystify automated risk classifications.
6. **Accuracy-Latency Optimization:** Analyze the operational trade-offs between 
 performance, execution speed, and model interpretability under real-time banking latency constraints.

---

## Regulatory Compliance Context
This framework is architected to address strict algorithmic governance mandates established by 
modern financial regulators and data protection laws:
* **GDPR Article 22 (Right to Explanation):** Ensuring that automated decisions that significantly
 impact consumers (such as blocking or declining account access) can be mathematically and textually e
 explained to the end-user.
* **The EU AI Act:** Categorizing financial risk assessment algorithms under high-governance 
transparency tiers, mandating robust audit trails and human-in-the-loop oversight capabilities.

---

## Dataset Schema
The project utilizes a comprehensive, synthetic banking transaction dataset containing 10,000 
unique records and 20 behavioral features. The variables provide a deep behavioral footprint well-suited for 
XAI calculations:

- **Transaction Metadata:** `transaction_amount`, `transaction_time_hour`, `exchange_rate`, `transaction_fees`.
- **Cyber-Behavioral Attributes:** `login_attempts`, `authentication_methods` 
(e.g., biometric vs. legacy pass), `device_risk_score`.
- **Account Profiles:** `account_age_days`, `failed_transactions_last_30d`, `transfer_frequency`.
- **Geographic Integrity:** `geographic_distance`, `location_mismatch_indicators`, `suspicious_ip_detection`.

### Target Variable
`fraud_flag`
* `0` = Legitimate Transaction (Majority Class)
* `1` = Fraudulent Transaction (Minority Class)

---

## Core Technologies
* **Language Environment:** Python 3.11+
* **Data Processing & Engineering:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE)
* **Predictive ML Core:** XGBoost, LightGBM
* **Explainable AI Engines:** SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-agnostic Explanations)
* **Visualization Layer:** Matplotlib, Seaborn
* **API Delivery Gateway:** FastAPI

---

## Machine Learning & Benchmark Pipeline
To comprehensively evaluate the trade-offs between model accuracy and interpretability, 
the framework benchmarks four distinct algorithms:

1. **Logistic Regression:** Serves as a baseline statistical model with low complexity and full linear interpretability.
2. **Decision Trees:** Provides a transparent, non-linear baseline showcasing inherent rule-based routing.
3. **Random Forest:** Introduces bagging ensemble complexity, boosting accuracy while masking inner decision logic.
4. **XGBoost & LightGBM:** Deploys state-of-the-art gradient boosting. LightGBM is explicitly leveraged to 
analyze computational optimizations and minimize the execution latency added by post-hoc XAI computations on 
real-time transaction rails.

---

## Explainable AI Methods
* **Global Interpretability (SHAP):** Grounded in cooperative game theory to map out overall 
network feature importances, showing compliance officers which variables consistently dictate 
risk parameters across the entire system.
* **Local Interpretability (LIME):** Generates instance-level perturbations around 
anomalous scores to translate exact model reasoning for unique, individual transaction alerts in real-time.

---

## Evaluation Metrics
Because standard accuracy metrics fail in severely skewed fraud datasets, performance will 
be judged on structural capability:
* **Confusion Matrix:** True Positives, False Positives, False Negatives, True Negatives.
* **Precision & Recall (F1-Score):** Measuring the exact truth rate of fraud alerts vs. 
the system's ability to capture all active threats.
* **ROC-AUC Score:** Tracking global classifier thresholds.
* **PR-AUC (Precision-Recall Area Under Curve):** The primary benchmark metric used to 
confirm model robustness on highly imbalanced minority-class classification data.

---

app/
│
├── backend/
│   ├── main.py          # FastAPI application initialization
│   └── inference.py     # Loads LightGBM/XGBoost models & calculates LIME/SHAP
│
└── frontend/
    └── dashboard.py     # Streamlit script displaying UI components & interactive charts
    

## Project Directory Structure
```text
data/
    raw/                       # Original Kaggle banking dataset
    processed/                 # Cleaned, encoded, and scaled arrays

notebooks/
    01_data_understanding.ipynb # Exploratory Data Analysis & Anomaly Isolation
    02_data_preprocessing.ipynb # SMOTE balancing and Feature Engineering
    03_model_training.ipynb     # Algorithmic Training, Hyperparameter tuning, and Latency Benchmarking
    04_explainability_analysis.ipynb # Local LIME perturbations & Global SHAP calculations

src/
    preprocessing.py           # Scaling, encoding, and train-test splits
    feature_engineering.py     # Constructing custom velocity & interaction features
    train.py                   # Custom script for ensemble pipeline compilation
    evaluate.py                # Compiling confusion matrices, PR-AUC, and loss curves
    explainability.py          # Serializing SHAP and LIME explainer objects

models/                        # Storing trained pickle / joblib serialization states

reports/                       # Storing visual validation plots and metric sheets

app/
    main.py                    # FastAPI initialization and route configurations
    inference.py               # Processing transactional payloads to generate risk arrays

README.md                      # Core Project Documentation
requirements.txt               # Unified library dependency list