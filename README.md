# 🚀 Hyperparameter Tuning Visualizer

An interactive Streamlit dashboard to visualize how different hyperparameters affect model performance using **Random Forest** — supports both classification and regression tasks.

> No manual preprocessing needed — just plug and play!

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## 🧠 Features

- 🔍 Auto-detects task type (classification or regression)
- ⚙️ Choose between default training or **GridSearchCV** tuning
- 📊 Displays metrics — Accuracy, RMSE, R², ROC Curve, Classification Report & Probability Histogram
- 📋 Real-time logs of all model training events
- 🤖 Auto handles encoding and missing value removal

---

## 📁 Dataset Requirements

Upload any `.csv` file that meets the following:

| ✅ Required | ❌ Avoid |
|---|---|
| Structured tabular format | Mixed-type columns (e.g. "355 hp") |
| One clearly defined target column | Embedded units/symbols (%, ₹, °C) |
| Numeric or categorical features | Free-text or unstructured columns |

> The app automatically handles encoding and missing value removal — no dummy columns needed!

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Roadmap
- [ ] Support for more models (XGBoost, SVM)
- [ ] Feature importance visualization
- [ ] Download tuned model as `.pkl`
