# Technical Documentation

## Project Overview

**Real-Time Loan Default Risk Prediction Platform with Automated MLOps Pipeline** is a production-style
machine learning system that predicts the probability a credit card customer will default on their
payment next month. It was built to demonstrate a complete, end-to-end MLOps lifecycle: from raw data
ingestion through training, deployment, real-time serving, monitoring, and automated retraining —
mirroring how a production AI system would be developed and maintained in an enterprise environment.

## Business Problem

Lenders need to assess credit risk before extending or renewing credit. Manually reviewing every
applicant does not scale, and a static model that is never updated becomes less accurate as customer
behavior changes over time (concept drift). This platform solves both problems: it serves real-time
risk predictions via an API, and it continuously monitors incoming data for drift, automatically
retraining and promoting a new model when the data distribution shifts meaningfully.

## Objectives

- Serve real-time default-risk predictions through a REST API
- Track every training run's parameters and metrics for reproducibility (MLflow)
- Detect data drift in incoming prediction traffic (Evidently)
- Automatically retrain and promote a new model when drift is detected, without manual intervention
- Package the application in Docker for consistent, portable deployment
- Automate testing and validation on every code change (CI/CD via GitHub Actions)
- Provide a human-readable monitoring dashboard (Streamlit)

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Dataset | UCI Default of Credit Card Clients (ID 350, via `ucimlrepo`) |
| ML | scikit-learn (Logistic Regression, Random Forest) |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Drift Monitoring | Evidently |
| Dashboard | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Deployment | ngrok tunnel (see [Deployment](#deployment) for details) |

## Dataset

The UCI "Default of Credit Card Clients" dataset contains 30,000 records of Taiwanese credit card
holders, with 23 raw features covering demographic information (credit limit, sex, education,
marriage, age) and 6 months of repayment history (repayment status, bill amount, and payment amount
for each of the last 6 months). The target variable, renamed `default`, indicates whether the customer
defaulted on their payment the following month.

The dataset's raw column names (`X1`–`X23`) are renamed to descriptive names (`LIMIT_BAL`, `SEX`,
`EDUCATION`, `PAY_0`...`PAY_6`, `BILL_AMT1`...`BILL_AMT6`, `PAY_AMT1`...`PAY_AMT6`) immediately after
loading, so every downstream stage works with readable feature names.

## Architecture
UCI Dataset
|
v
Data Loading (src/data_loader.py)
|
v
Preprocessing (src/preprocess.py) — cleaning, category fixing, train/test split
|
v
Feature Engineering (src/features.py) — payment history aggregates
|
v
Model Training (src/train.py) — Logistic Regression + Random Forest, MLflow logging
|
v
Evaluation (src/evaluate.py) — confusion matrix, ROC curve, feature importance
|
v
Best Model Saved (models/best_model.pkl) + Metadata (models/model_metadata.json)
|
v
FastAPI Real-Time Prediction (api/main.py) — /health, /predict
|
v
Prediction Logging (src/monitoring.py) — data/monitoring/predictions.csv
|
v
Evidently Drift Monitoring (src/monitoring.py)
|
+--> No Drift --> Keep Current Model
|
`--> Drift Detected --> Retraining (src/retrain.py) --> Evaluate Candidate
|
Candidate ROC-AUC >= Current?
/
Yes No
| |
Promote Keep Current

GitHub --> GitHub Actions (CI) --> Tests --> Docker Build --> Deployment


A visual version of this diagram is provided in `docs/architecture.png`.

## Folder Structure

loan-default-mlops-platform/
├── data/
│ ├── raw/ # Raw dataset from UCI
│ ├── processed/ # Train/test splits
│ ├── reference/ # Drift reference snapshot (training data)
│ └── monitoring/ # Logged live predictions
├── models/ # Trained model, metrics, metadata
├── reports/
│ ├── figures/ # Confusion matrix, ROC curve, feature importance
│ └── drift/ # Drift HTML/JSON reports, retrain summary
├── notebooks/ # Exploratory data analysis
├── src/ # Core pipeline modules
├── api/ # FastAPI application
├── dashboard/ # Streamlit monitoring dashboard
├── tests/ # Pytest suite
├── .github/workflows/ # CI and scheduled retraining workflows
├── docs/ # This documentation + architecture diagram
├── Dockerfile
└── requirements.txt


## How to Run Locally

**1. Set up the environment:**

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


**2. Run the full training pipeline** (data loading → preprocessing → training → MLflow logging → evaluation → reference dataset):

python -m src.pipeline


**3. Start the API:**

uvicorn api.main:app --reload

Visit `http://127.0.0.1:8000/` for the landing page, `/docs` for interactive Swagger testing.

**4. Start the monitoring dashboard:**

streamlit run dashboard/app.py


**5. View MLflow experiment tracking:**

mlflow ui


**6. Run drift detection manually:**

python -m src.monitoring


**7. Run the retraining workflow manually:**

python -m src.retrain


**8. Run tests:**

pytest -v


## API Reference

### `GET /health`
Returns API and model status.
```json
{"status": "ok", "model_loaded": true, "model_version": 5}
```

### `POST /predict`
Accepts the 23 raw applicant features (see `api/schemas.py` for the full field list and validation
rules) and returns:
```json
{"prediction": 0, "probability": 0.1478, "risk_level": "Low", "model_version": 5}
```
- `prediction`: 0 (no default) or 1 (default)
- `probability`: model's predicted probability of default
- `risk_level`: Low (< 0.30), Medium (0.30–0.60), or High (> 0.60)

Engineered features (payment averages, credit utilization, payment trend) are computed internally from
the raw inputs — they are never required as request fields, and the exact same feature engineering
logic (`src/features.py`) is used at both training and inference time to avoid training-serving skew.

## Experiment Tracking (MLflow)

Every training run — whether from `src/train.py`, the full pipeline, or an automated retraining run —
logs to the `Loan_Default_Risk_Prediction` MLflow experiment: model name, hyperparameters, all 5
evaluation metrics, and the fitted model artifact. Retraining runs are additionally tagged
`trigger=retrain` so they can be filtered separately from regular training runs in the MLflow UI.

**Limitation:** MLflow tracking in this project uses a local SQLite/file store (`mlruns/`, `mlflow.db`),
which is excluded from version control and does not persist across container restarts or redeploys. In
a production setting, this would be replaced with a hosted MLflow tracking server backed by persistent
storage.

## Monitoring and Drift Detection

Every successful `/predict` call is logged to `data/monitoring/predictions.csv` (timestamp, raw
inputs, prediction, probability, risk level, model version), without blocking or slowing the API
response if logging fails.

Drift detection (`src/monitoring.py`) compares this live prediction log against a reference snapshot
of the training data (`data/reference/reference_data.csv`), using Evidently's `DataDriftPreset`. Only
raw feature columns common to both datasets are compared. If fewer than 30 predictions have been
logged, drift detection is skipped gracefully rather than run on insufficient data.

**Known limitation:** statistical drift tests on low-cardinality categorical features (e.g. `SEX`,
`EDUCATION`, `PAY_0`) are prone to false positives at small sample sizes — this was observed directly
during testing, where drift was flagged even on data resampled from the training distribution itself,
due to sampling noise on categorical columns with few unique values. This is a known characteristic of
statistical drift detection at small sample sizes, not a defect in the implementation, and is a
realistic constraint in early-stage production monitoring before enough traffic accumulates.

## Automated Retraining

`src/retrain.py` implements the full drift-triggered retraining workflow:
1. Run drift detection
2. If no drift, exit without retraining
3. If drift detected, retrain candidate model(s) using available labelled data
4. Compare candidate ROC-AUC against the current production model's ROC-AUC
5. Promote the candidate only if it meets or exceeds the current model's ROC-AUC
6. The production model file is never overwritten before this comparison completes
7. Log the retraining run to MLflow, tagged `trigger=retrain`
8. Write a summary to `reports/retrain_summary.json`

**Important documented limitation:** real-time prediction requests do not include the true future
default outcome — that information does not exist at inference time. Genuinely new labelled data is
therefore not available for retraining in this project's timeframe. Retraining instead reuses the
existing labelled training data as a stand-in for "updated" data. This demonstrates the complete
retraining mechanism (drift detection → retraining → evaluation → comparison → promotion) end-to-end,
as a realistic simulation of the production workflow, without claiming access to labels that would not
genuinely exist in this real-time scenario. This was validated directly: a retraining run correctly
detected drift, retrained a candidate, compared it against the current model (an exact tie, since the
same data and random seed were used), and correctly promoted it — confirming the `>=` promotion
boundary works correctly, not only the "clearly better" case.

Retraining is available both as a manual command (`python -m src.retrain`) and as a GitHub Actions
workflow (`.github/workflows/retrain.yml`), triggerable on-demand (`workflow_dispatch`) or on a weekly
schedule.

## Model Performance

Two candidate models are trained and compared on every run: Logistic Regression and Random Forest.
The best model is selected by ROC-AUC. Current results (see `models/metrics.json` for exact, live
figures — not reproduced statically here to avoid stale numbers):

- Random Forest was selected as the best model, with ROC-AUC in the range of ~0.76–0.77 across training
  runs — a realistic, credible result for this dataset and problem (not overfit, not underperforming).
- Full confusion matrix, ROC curve, and feature importance plots are generated by `src/evaluate.py` and
  saved to `reports/figures/`, and are also displayed in the Streamlit dashboard's Model Performance tab.

## Docker

The application is containerized via a single `Dockerfile` at the project root. Library versions
(`pandas`, `numpy`, `scikit-learn`, `joblib`) are explicitly pinned in `requirements.txt` to match the
versions the model was actually trained with, avoiding pickle-compatibility issues between the local
training environment and the container. The container reads its listening port from the `PORT`
environment variable (defaulting to 8000 when unset), so it works both locally and on cloud platforms
that assign ports dynamically.

docker build -t loan-default-api .
docker run -p 8000:8000 loan-default-api


## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request targeting `main`: checks out the code,
sets up Python 3.13, installs pinned dependencies, and runs the full Pytest suite (18 tests covering
preprocessing, feature engineering, model behavior, and the API).

`.github/workflows/retrain.yml` runs the retraining workflow on a weekly schedule or on-demand via
manual trigger.

Branch protection is enabled on `main`: all changes must go through a pull request from `develop` or a
feature/fix branch, and CI must pass before merging.

## Deployment

**Note on deployment approach:** initial attempts were made to deploy to Render and other free-tier
cloud platforms; each ultimately required credit card verification for account/service creation, even
on their advertised free tiers. Given the project timeline, the application is instead deployed using
an ngrok tunnel, which exposes the locally-running Docker container to a public HTTPS URL without any
payment information. This is documented here transparently as the deployment mechanism actually used,
rather than presented as a persistent cloud deployment it is not. The container itself, its
Dockerfile, and its port-binding logic are written to be cloud-deployment-ready (respecting the `PORT`
environment variable, for example) should a card-free or approved-payment cloud platform become
available.

## Known Limitations

- MLflow tracking data and prediction/monitoring logs use local file storage, which does not persist
  across container restarts — acceptable for this demonstration, but would require a persistent
  backing store (database, hosted MLflow server) in a genuine production deployment.
- Retraining uses simulated/reused labelled data rather than genuinely new real-time labels, since true
  outcomes are not available at inference time in this scenario (see [Automated Retraining](#automated-retraining)).
- Drift detection can produce false positives on categorical features at small sample sizes.
- The current deployment (ngrok tunnel) is not a persistent, always-on cloud service; it requires the
  host machine and tunnel process to remain running.

## Future Improvements

- Deploy to a genuine persistent cloud platform (Render, Railway, or similar) once payment verification
  is acceptable, or once a genuinely card-free platform is identified
- Replace local MLflow file storage with a hosted tracking server
- Persist prediction logs and drift history to a database rather than local CSV/JSON files
- Collect genuine labelled outcomes over time (e.g. via a feedback loop) to enable truly real-time
  retraining rather than simulated data
- Add authentication/rate-limiting to the public API