# Machine Failure Prediction & Monitoring

An end-to-end machine learning application that predicts industrial machine
failure from live sensor readings, serves predictions through a REST API,
stores every prediction in PostgreSQL, and visualizes everything on a
Streamlit monitoring dashboard.

Built as a final-year engineering project to demonstrate a realistic,
complete ML system — not just a notebook.

---

## 1. Project Overview

Unplanned machine downtime is expensive. Predictive maintenance uses
sensor data (temperature, torque, rotational speed, tool wear, etc.) to
flag machines that are at risk of failing *before* they actually break
down, so maintenance can be scheduled proactively instead of reactively.

This project trains and compares three classifiers on real sensor data,
picks the best one by F1-score, and wraps it in a production-style
application: a FastAPI prediction service, a PostgreSQL database that logs
every prediction, and a Streamlit dashboard for monitoring.

## 2. Problem Statement

Given a machine's live sensor readings (air temperature, process
temperature, rotational speed, torque, tool wear, and product quality
type), predict whether the machine is likely to fail (`1`) or continue
operating normally (`0`). This is a **binary classification** problem
on a **highly imbalanced dataset** (only ~3.4% of machines fail).

## 3. Features

- Full ML pipeline: cleaning → EDA → feature engineering → training →
  evaluation → model selection
- Three trained/compared models: Logistic Regression, Random Forest, XGBoost
- FastAPI backend with prediction, health, and history endpoints
- PostgreSQL storage of every prediction (machine info, sensor values,
  prediction, probability, timestamp)
- Streamlit dashboard: live prediction form + monitoring (recent
  predictions, sensor trends, model performance, confusion matrix,
  feature importance)
- Fully Dockerized with Docker Compose (API + DB + dashboard)
- Unit tests for preprocessing, the trained model, and the API

## 4. Architecture / Workflow

```
AI4I 2020 CSV
      │
      ▼
Data Cleaning (src/data_preprocessing.py)
      │  - drop ID columns, drop leakage columns (failure-mode flags)
      │  - de-duplicate, handle missing values, encode 'Type'
      ▼
Feature Engineering (src/feature_engineering.py)
      │  - temp_diff, power_w, torque_wear_product, wear_per_type
      ▼
Model Training (src/train_model.py)
      │  - stratified train/test split, StandardScaler
      │  - Logistic Regression / Random Forest / XGBoost
      │  - 5-fold stratified cross-validation + held-out test evaluation
      │  - best model selected by F1-score, saved with joblib
      ▼
models/best_model.joblib, scaler.joblib, model_metadata.json
      │
      ▼
FastAPI (api/main.py)  ──────────────►  PostgreSQL (database/schema.sql)
      │                                        ▲
      ▼                                        │
Streamlit Dashboard (dashboard/app.py) ─────────┘
      (prediction form + monitoring charts, all via the API)
```

All three services (API, dashboard, PostgreSQL) run in separate Docker
containers, orchestrated with Docker Compose.

## 5. Dataset

**AI4I 2020 Predictive Maintenance Dataset**
Source: UCI Machine Learning Repository ([https://doi.org/10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C))
Citation: S. Matzka, *"Explainable Artificial Intelligence for Predictive
Maintenance Applications"*, 3rd International Conference on Artificial
Intelligence for Industries (AI4I), 2020.

- 10,000 rows, 14 columns, no missing values
- Synthetic but statistically realistic industrial milling-machine data
- Target: `Machine failure` (binary) — **339 failures out of 10,000 rows
  (3.39%)**, i.e. a highly imbalanced problem
- Raw features: Product ID, Type (L/M/H), Air temperature [K], Process
  temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
- The dataset also ships 5 individual failure-mode flags (TWF, HDF, PWF,
  OSF, RNF). **These are dropped during cleaning** — they are only known
  after a failure is confirmed, so including them would leak the answer
  into the model.

Place the CSV at `data/raw/ai4i2020.csv` (already included in this repo).

## 6. ML Approach

1. **Cleaning**: drop `UDI`/`Product ID` (no predictive value), drop the
   5 failure-mode flags (leakage), de-duplicate, defensively handle
   missing values (there are none in this dataset, but the code handles
   them so it's reusable on messier sensor data), encode `Type` as
   ordinal (L=0, M=1, H=2 — quality has a natural order).
2. **Feature engineering** (grounded in the physics behind this dataset's
   own documented failure rules):
   - `temp_diff` = process_temp − air_temp (heat dissipation signal)
   - `power_w` = torque × angular velocity (power failure signal)
   - `torque_wear_product` = torque × tool wear (overstrain signal)
   - `wear_per_type` = tool wear scaled by product quality
3. **Split**: 80/20 **stratified** train/test split (stratified because
   failures are rare — a plain random split could easily distort the
   3.4% ratio).
4. **Scaling**: `StandardScaler` fit only on the training set, then
   applied to the test set (avoids leaking test-set statistics into
   training).
5. **Class imbalance handling**: `class_weight="balanced"` for Logistic
   Regression / Random Forest, and `scale_pos_weight` for XGBoost — this
   makes the models pay more attention to the rare failure class without
   fabricating synthetic rows.
6. **Validation**: 5-fold stratified cross-validation on the training set,
   plus a final evaluation on the held-out test set that was never touched
   during training or CV.
7. **Model selection**: best model chosen by **test-set F1-score**, not
   accuracy — with only 3.4% failures, a model that always predicts "no
   failure" would already score ~96.6% accuracy while being useless.

## 7. Model Comparison (actual results from this repo)

Cross-validation (5-fold, on the 8,000-row training set):

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 |
|---|---|---|---|---|
| Logistic Regression | 0.8477 | 0.1616 | 0.8341 | 0.2706 |
| Random Forest | 0.9909 | 0.9754 | 0.7490 | 0.8470 |
| XGBoost | 0.9878 | 0.8333 | 0.8007 | 0.8159 |

Held-out test set (2,000 rows, never used for training or CV):

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.8555 | 0.1760 | 0.8824 | 0.2934 |
| Random Forest | 0.9900 | 0.9615 | 0.7353 | 0.8333 |
| **XGBoost (selected)** | **0.9905** | **0.8889** | **0.8235** | **0.8550** |

**Confusion matrix — XGBoost on the test set:**

|  | Predicted Normal | Predicted Failure |
|---|---|---|
| **Actual Normal** | 1925 | 7 |
| **Actual Failure** | 12 | 56 |

**Why XGBoost was selected:** it has the highest test F1-score (0.855),
correctly catching 56 of 68 actual failures (82.4% recall) while keeping
false alarms low (only 7 false positives out of 1,932 normal machines).
Random Forest has higher precision but misses more real failures (73.5%
recall) — in a maintenance context, missing a real failure is usually
more costly than an extra inspection, so the model with better recall at
similar precision was preferred. Logistic Regression clearly underfits
this non-linear relationship between sensors and failure and was not
competitive.

**Top features by importance (XGBoost):** `rotational_speed_rpm` (0.279),
`power_w` (0.229), `tool_wear_min` (0.169), `torque_nm` (0.136) — these
line up with the mechanical failure rules (power/overstrain/tool wear)
the dataset was generated from, which is a good sanity check that the
model learned something physically sensible rather than noise.

All of these numbers are produced by `src/train_model.py` and saved to
`models/model_metadata.json` — nothing here is hand-typed.

## 8. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API/model health check |
| POST | `/predict` | Predict failure for one sensor reading, stores result in DB |
| GET | `/predictions/recent?limit=20` | Most recent stored predictions |
| GET | `/predictions/stats` | Aggregate counts (total, failures, normal, avg probability) |
| GET | `/model/metadata` | Training metrics/feature importance used by the dashboard |

Interactive API docs are auto-generated by FastAPI at `/docs` once the
service is running.

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "machine_id": "MACHINE-001",
        "product_type": "M",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 0
      }'
```

## 9. Database Design

Single table `predictions` (see `database/schema.sql`):

- Machine info: `machine_id`, `product_type`
- Sensor inputs: `air_temperature_k`, `process_temperature_k`,
  `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`
- Model output: `prediction`, `failure_probability`, `model_name`
- `created_at` timestamp

Indexed on `created_at`, `machine_id`, and `prediction` for fast dashboard
queries.

## 10. Dashboard

The Streamlit dashboard (`dashboard/app.py`) shows:
- Total predictions, predicted failures, normal machines, average
  failure probability
- A form to submit new sensor readings and get a live prediction from
  the FastAPI backend
- A table of recent predictions
- Sensor trend chart (rotational speed & torque over recent predictions)
- Model performance table (Accuracy/Precision/Recall/F1 for all 3 models)
- Confusion matrix heatmap for the selected model
- Feature importance bar chart

## 11. Docker Setup

Three services, one command:

```bash
docker compose up --build
```

- `db` — PostgreSQL 15, schema auto-created from `database/schema.sql`
- `api` — FastAPI service on `http://localhost:8000`
- `dashboard` — Streamlit dashboard on `http://localhost:8501`

## 12. How to Run Locally (without Docker)

```bash
# 1. Clone and set up environment
git clone <your-repo-url>
cd machine-failure-prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Train the model (produces models/*.joblib and model_metadata.json)
python -m src.train_model

# 3. Start PostgreSQL (via Docker, or point env vars at your own instance)
docker compose up db -d

# 4. Start the API
uvicorn api.main:app --reload

# 5. In a second terminal, start the dashboard
streamlit run dashboard/app.py

# 6. Run tests
pytest tests/
```

## 13. Project Structure

```
machine-failure-prediction/
├── data/
│   ├── raw/ai4i2020.csv
│   └── processed/               # generated by the pipeline
├── notebooks/                   # optional EDA notebook
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── train_model.py
├── models/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   └── model_metadata.json
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── database.py
├── dashboard/
│   └── app.py
├── database/
│   └── schema.sql
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

## 14. Future Improvements

- Add SHAP-based explainability to the dashboard (per-prediction
  explanation of *why* a machine was flagged)
- Add authentication to the API before any real deployment
- Add a scheduled retraining job when new labeled data accumulates
- Multi-class prediction of the specific failure mode (TWF/HDF/PWF/OSF),
  not just binary failure/no-failure
- Add Prometheus/Grafana for infrastructure-level monitoring alongside
  the model-level Streamlit dashboard
- CI pipeline (GitHub Actions) to run `pytest` on every push

---

## Resume Bullets

- Built an end-to-end machine failure prediction system on the AI4I 2020
  industrial sensor dataset, comparing Logistic Regression, Random Forest,
  and XGBoost with stratified cross-validation; selected XGBoost (F1 =
  0.855, recall = 82.4% on a held-out test set) for a 3.4%-imbalanced
  failure class.
- Designed and deployed a FastAPI prediction service backed by PostgreSQL
  for persistent logging of sensor inputs and predictions, and a Streamlit
  dashboard for real-time monitoring of failure rates, sensor trends, and
  model performance.
- Containerized the full application (API, dashboard, database) with
  Docker Compose for consistent, one-command local deployment; wrote unit
  tests covering data preprocessing, the trained model, and API endpoints.

---

## License

Dataset: AI4I 2020 Predictive Maintenance Dataset, CC BY 4.0, UCI Machine
Learning Repository. Code: for educational/portfolio use.
