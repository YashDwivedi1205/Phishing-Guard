# 🛡️ PhishGuard — Phishing Website Detector

An end-to-end machine learning system that classifies websites as **safe** or **phishing** based on 30 structural and behavioral features — built from data ingestion all the way to a live, deployed web application.

**[🔗 Live Demo](https://phishing-detector-ml-ujyg.onrender.com/)** &nbsp;|&nbsp; **[📂 Source Code](https://github.com/YashDwivedi1205/Phishing-Guard)**

---

## Overview

PhishGuard takes a CSV of website features and returns a prediction — *safe* or *phishing* — along with a confidence score for each row. It's built as a modular, production-style ML pipeline rather than a single notebook, following patterns used in real-world MLOps systems.

| | |
|---|---|
| **Test Accuracy** | ~95% |
| **Algorithms compared** | XGBoost, Logistic Regression, Gaussian Naive Bayes |
| **Tuning** | GridSearchCV (5-fold cross-validation) |
| **Dataset** | 11,000+ labeled websites, 30 features (UCI Phishing Websites Dataset) |

---

## Features

- **End-to-end ML pipeline** — data ingestion, schema validation, transformation, and training as independent, testable components
- **Multi-algorithm model selection** — trains and compares multiple algorithms, then fine-tunes the best one
- **MongoDB Atlas** as the data source, with a dedicated data-access layer
- **Batch predictions** via a web UI — upload a CSV, get results with confidence scores
- **Cloud model versioning** via Hugging Face Hub
- **Custom logging & exception handling** throughout, for real debuggability
- **Dockerized and deployed live** on Render

---

## Architecture

```
MongoDB Atlas (raw data)
        │
        ▼
 Data Ingestion  ──▶  Data Validation  ──▶  Data Transformation  ──▶  Model Training
   (fetch, split)      (schema check)        (preprocessing)          (multi-algo + tuning)
                                                                              │
                                                                              ▼
                                                                     model.pkl (Hugging Face Hub)
                                                                              │
                                                                              ▼
                                                                    Flask App (predict_pipeline)
                                                                              │
                                                                              ▼
                                                                       Web UI (Docker → Render)
```

---

## Project Structure

```
Phishing-Guard/
├── src/
│   ├── components/          # data_ingestion, data_validation, data_transformation, model_trainer
│   ├── configuration/       # MongoDB connection setup
│   ├── constant/            # centralized config values
│   ├── data_access/         # MongoDB → DataFrame layer
│   ├── entity/               # artifact dataclasses passed between stages
│   ├── pipeline/             # train_pipeline.py, predict_pipeline.py
│   ├── utils/                 # shared helpers (save/load objects, YAML, HF upload/download)
│   ├── exception.py          # custom exception with file + line number tracing
│   └── logger.py             # timestamped file logging
├── config/
│   ├── schema.yaml           # expected column schema for validation
│   └── model.yaml            # hyperparameter search grids per algorithm
├── templates/                 # prediction.html, results.html
├── static/css/                # custom UI styling
├── app.py                     # Flask application
├── Dockerfile
├── requirements.txt
└── setup.py
```

---

## How It Works

1. **Data Ingestion** — fetches the dataset from MongoDB Atlas, removes duplicates, and splits it into train/test sets.
2. **Data Validation** — checks the data against `schema.yaml` (column count and names) before it's allowed to proceed.
3. **Data Transformation** — applies preprocessing and saves the fitted preprocessor as a reusable object.
4. **Model Training** — trains Logistic Regression, Gaussian Naive Bayes, and XGBoost, selects the best performer, and fine-tunes it with `GridSearchCV` using the grids defined in `model.yaml`.
5. **Deployment** — the final model is pushed to Hugging Face Hub and pulled at inference time by the Flask app if not already cached locally.
6. **Prediction** — a user uploads a CSV of website features through the web UI; the app returns each row's classification with a confidence percentage, viewable on-screen and downloadable as CSV.

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/YashDwivedi1205/Phishing-Guard.git
cd Phishing-Guard

# Set up environment
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Add your own .env file with:
# MONGODB_URL=your_mongodb_connection_string
# HUGGINGFACE_TOKEN=your_huggingface_token

# Train the model
python src/pipeline/train_pipeline.py

# Run the app
python app.py
```

Then open `http://localhost:10000` in your browser.

---

## Tech Stack

**ML/Data:** Python, scikit-learn, XGBoost, Pandas, NumPy, GridSearchCV
**Backend/Data:** Flask, MongoDB Atlas, PyMongo
**Deployment:** Docker, Render, Hugging Face Hub
**Frontend:** HTML, CSS, JavaScript

---

## Dataset

This project uses the [UCI Phishing Websites Dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites), which contains 30 lexical, host-based, and content-based features extracted from URLs and web pages.

---

## Future Improvements

- [ ] Add data drift monitoring between training and inference data
- [ ] Add authentication for the `/train` endpoint
- [ ] Support single-URL prediction (not just batch CSV upload)
- [ ] Add automated CI/CD pipeline via GitHub Actions

---

## Author

**Yash Dwivedi**
[LinkedIn](https://www.linkedin.com/in/yash) · [GitHub](https://github.com/YashDwivedi1205)