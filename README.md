# FraudLens — Financial Fraud Detection & Transaction Risk Analytics

A ready-to-run Python Data Science mini project. It includes a generated synthetic transaction dataset, exploratory dashboard, model training pipeline, model comparison and single-transaction risk scoring.

## Important dataset note
`data/transactions.csv` is **synthetic demonstration data generated for this project**. It is not real bank/customer data and must not be represented as a real-world banking dataset. Model metrics on synthetic data demonstrate the workflow only; they do not establish real-world fraud-detection performance.

## Features
- 5,000 generated transaction records
- Data exploration and visual analytics
- Preprocessing pipeline: median imputation, scaling and one-hot encoding
- Model comparison: Logistic Regression, Random Forest and HistGradientBoosting
- Metrics: precision, recall, F1, ROC-AUC and average precision
- Streamlit dashboard with single-transaction scoring

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract this project folder.
3. Open Command Prompt in the project folder.
4. Create and activate a virtual environment:

   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Train and save the model:

   ```bat
   python train_model.py
   ```

7. Start the dashboard:

   ```bat
   streamlit run app.py
   ```

8. Open the local address printed in the terminal (usually http://localhost:8501).

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Project structure
```text
financial_fraud_detection_project/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── transactions.csv
├── models/                 # model file created after training
├── outputs/                # metrics created after training
├── documentation/
│   └── Project_Report.docx
└── presentation/
    └── Project_Presentation.pptx
```

## Suggested mini-project workflow
1. Understand the columns and target variable.
2. Explore class balance and transaction distributions.
3. Train and compare the three models.
4. Discuss precision/recall trade-offs and false positives/false negatives.
5. Demonstrate the dashboard and test a few sample transactions.

## Limitations and responsible use
- Synthetic data does not capture the complexity, drift, adversarial behaviour or operational constraints of real payment systems.
- A 0.5 threshold is a demonstration default, not a production threshold.
- Never use this prototype to block, approve or accuse real customers.
- A real deployment requires authorized, representative data; privacy/security review; threshold calibration; monitoring; explainability; human review; and compliance approval.
