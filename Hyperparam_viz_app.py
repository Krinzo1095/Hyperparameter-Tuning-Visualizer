import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,mean_squared_error, r2_score,roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

st.set_page_config(layout="wide")
st.title("Universal ML Model Trainer 🧠")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

def auto_preprocess(df):
    df = df.copy()
    df.dropna(axis=0, inplace=True)
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    df = auto_preprocess(df)
    target_column = st.selectbox("Select the target column", df.columns)

    task_type = st.selectbox("Choose Task Type", ["classification", "regression"])

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    use_grid = st.checkbox("Use GridSearchCV for Hyperparameter Tuning")

    if not use_grid:
        st.write("## Random Forest Parameters")
        n_estimators = st.slider("Number of Trees", 10, 500, 100)
        max_depth = st.slider("Max Depth", 1, 50, 10)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42) if task_type == "classification" \
                else RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    else:
        st.write("## GridSearch Parameters (using predefined ranges)")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20]
        }
        base_model = RandomForestClassifier(random_state=42) if task_type == "classification" else RandomForestRegressor(random_state=42)
        model = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy' if task_type == "classification" else 'r2')

    if st.button("Train Model"):
        model.fit(X_train, y_train)

        if use_grid:
            best_model = model.best_estimator_
            st.success("Best Parameters Found:")
            st.json(model.best_params_)
        else:
            best_model = model

        y_pred = best_model.predict(X_test)

        if task_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Model Accuracy: {acc:.4f}")

            # Classification Report
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

            # ROC Curve
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)

                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax_roc.set_title("ROC Curve")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend()
                st.pyplot(fig_roc)

                # Probability Histogram
                fig_hist, ax_hist = plt.subplots()
                ax_hist.hist(y_proba, bins=20, color='skyblue', edgecolor='black')
                ax_hist.set_title("Predicted Probabilities Histogram")
                ax_hist.set_xlabel("Probability of Class 1")
                ax_hist.set_ylabel("Count")
                st.pyplot(fig_hist)

        else:
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            st.success("Regression Model Trained Successfully!")
            st.markdown(f"**RMSE:** {rmse:.4f}")
            st.markdown(f"**R² Score:** {r2:.4f}")

  