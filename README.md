🚀 Hyperparameter-Tuning-Visualizer
An interactive Streamlit dashboard that allows users to upload their dataset and visualize how different hyperparameters affect model performance using Random Forest (for both classification and regression).
No manual preprocessing needed — just plug and play!

<br>
📁 Upload Guidelines – Compatible Dataset Requirements
To ensure smooth operation, your CSV dataset should meet the following criteria:

📊 Must be in structured tabular format (.csv)

🎯 Must have one clearly defined target column (for classification or regression tasks)

🔢 All feature columns should be either:

Purely numeric, or

Categorical (e.g., "Male", "Red", "Yes")

❌ Avoid:

Mixed-type columns (like "355 hp", "20 km/l")

Embedded units/symbols (e.g., %, ₹, °C)

🔧 No need for:

Manual preprocessing

Creating dummy columns
The app automatically handles encoding and missing value removal.

<br>
🧠 Features
Automatically detects task type (classification or regression)

Choose between default training or GridSearchCV tuning

Displays metrics: Accuracy, RMSE, R², ROC Curve, Classification Report, and Probability Histogram

Real-time logs of all model training events

