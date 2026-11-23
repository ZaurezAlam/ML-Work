# ==============================================================================
# Refactored Data Science & Machine Learning Portfolio / CV Reference Script
# Clean, modular, warning-free, and leakage-free implementations
# ==============================================================================

# ------------------------------------------------------------------------------
# 2. Data Collection and Cleaning
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import zscore

data = {
    "Name": ["Alice", "Bob", "Charlie", None, "Eve"],
    "Age": [25, None, 35, 28, 120],
    "Salary": [50000, 60000, None, 58000, 55000],
    "Department": ["HR", "Finance", "HR", "IT", "IT"],
}

df = pd.DataFrame(data)

# Step 1: Handle missing values without chained inplace warnings (pandas 3.0 safe)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
df = df.dropna(subset=["Name"]).copy()

# Step 2: Handle domain inconsistencies
df = df[df["Age"] <= 100].copy()

# Step 3: Detect and filter salary outliers using Z-score
salary_z = zscore(df["Salary"])
df = df[np.abs(salary_z) <= 3].copy()

# Step 4: Standardize text formatting
df["Name"] = df["Name"].str.strip().str.title()

print("=== Cleaned Employee Dataset ===")
print(df)


# ------------------------------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(TITANIC_URL)

print("\n=== Dataset Overview ===")
titanic.info()

print("\n=== Summary Statistics ===")
print(titanic.describe(include="all"))

print("\n=== Missing Values ===")
print(titanic.isnull().sum())

# Safe imputation
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])

# Univariate Analysis
plt.figure(figsize=(8, 5))
sns.histplot(titanic["Age"], kde=True, bins=30, color="steelblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=titanic, x="Sex", hue="Sex", palette="pastel", legend=False)
plt.title("Gender Distribution")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Bivariate Analysis
plt.figure(figsize=(8, 5))
sns.barplot(data=titanic, x="Sex", y="Survived", hue="Sex", palette="muted", legend=False)
plt.title("Survival Rate by Gender")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=titanic, x="Pclass", y="Survived", hue="Pclass", palette="coolwarm", legend=False)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = titanic.select_dtypes(include=["number"]).columns
sns.heatmap(titanic[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 4. Feature Engineering
# ------------------------------------------------------------------------------
titanic_fe = pd.read_csv(TITANIC_URL)
titanic_fe["FamilySize"] = titanic_fe["SibSp"] + titanic_fe["Parch"] + 1
titanic_fe["IsAlone"] = (titanic_fe["FamilySize"] == 1).astype(int)

print("\n=== Engineered Features Preview ===")
print(titanic_fe[["SibSp", "Parch", "FamilySize", "IsAlone"]].head())

plt.figure(figsize=(8, 5))
sns.barplot(data=titanic_fe, x="FamilySize", y="Survived", hue="FamilySize", palette="coolwarm", legend=False)
plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(data=titanic_fe, x="IsAlone", y="Survived", hue="IsAlone", palette="muted", legend=False)
plt.title("Survival Rate by Being Alone")
plt.xlabel("IsAlone")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 5. Data Visualization
# ------------------------------------------------------------------------------
titanic_viz = pd.read_csv(TITANIC_URL)
titanic_viz["Age"] = titanic_viz["Age"].fillna(titanic_viz["Age"].median())
titanic_viz["Embarked"] = titanic_viz["Embarked"].fillna(titanic_viz["Embarked"].mode()[0])

plt.figure(figsize=(8, 5))
sns.barplot(data=titanic_viz, x="Pclass", y="Survived", hue="Pclass", palette="coolwarm", legend=False)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=titanic_viz, x="Age", y="Fare", hue="Survived", palette="Set1", alpha=0.7)
plt.title("Age vs. Fare (Colored by Survival)")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
num_cols = titanic_viz.select_dtypes(include=["number"]).columns
sns.heatmap(titanic_viz[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 6. Statistical Analysis (Two-Sample T-Test & Chi-Square Test)
# ------------------------------------------------------------------------------
from scipy import stats

titanic_stats = pd.read_csv(TITANIC_URL)
male_survival = titanic_stats.loc[titanic_stats["Sex"] == "male", "Survived"]
female_survival = titanic_stats.loc[titanic_stats["Sex"] == "female", "Survived"]

t_stat, p_value = stats.ttest_ind(male_survival, female_survival, equal_var=False)
print(f"\n=== Statistical Hypothesis Test (Welch's t-test) ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4e}")

if p_value < 0.05:
    print("Reject the null hypothesis: Significant difference in survival rates between male and female passengers.")
else:
    print("Fail to reject the null hypothesis: No significant difference in survival rates.")


# ------------------------------------------------------------------------------
# 7, 8 & 9. Leakage-Free ML Pipeline: Training, Evaluation & GridSearchCV
# ------------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

titanic_ml = pd.read_csv(TITANIC_URL)
features = ["Pclass", "Age", "Fare", "Sex", "Embarked"]
X = titanic_ml[features]
y = titanic_ml["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = ["Pclass", "Age", "Fare"]
categorical_features = ["Sex", "Embarked"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ]
)

# Model 1: Logistic Regression Pipeline
logreg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500, random_state=42)),
])
logreg_pipeline.fit(X_train, y_train)
y_pred_logreg = logreg_pipeline.predict(X_test)

# Model 2: Baseline Decision Tree Pipeline
dtree_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42)),
])
dtree_pipeline.fit(X_train, y_train)
y_pred_dtree = dtree_pipeline.predict(X_test)


def print_classification_metrics(name, y_true, y_pred):
    print(f"\n=== {name} Metrics ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred))


print_classification_metrics("Logistic Regression", y_test, y_pred_logreg)
print_classification_metrics("Decision Tree (Baseline)", y_test, y_pred_dtree)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    "classifier__max_depth": [3, 5, 10, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__criterion": ["gini", "entropy"],
}

grid_search = GridSearchCV(
    estimator=dtree_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print("=== Tuned Decision Tree Results ===")
print("Best Hyperparameters:", grid_search.best_params_)
print(f"Best Cross-Validation F1 Score: {grid_search.best_score_:.4f}")

best_dtree = grid_search.best_estimator_
y_pred_best_dtree = best_dtree.predict(X_test)
print_classification_metrics("Optimized Decision Tree", y_test, y_pred_best_dtree)


# ------------------------------------------------------------------------------
# 10. Model Serialization & Flask REST API Deployment (Bug-Fixed)
# ------------------------------------------------------------------------------
# Note: Never hardcode ngrok auth tokens in source files; use environment variables.
import os
import pickle
from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression

X_train_reg = [[1], [2], [3], [4], [5]]
y_train_reg = [2, 4, 6, 8, 10]

reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

MODEL_PATH = "linear_regression_model.pkl"
with open(MODEL_PATH, "wb") as file:
    pickle.dump(reg_model, file)

app = Flask(__name__)

with open(MODEL_PATH, "rb") as file:
    loaded_model = pickle.load(file)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "input_value" not in data:
        return jsonify({"error": "Missing 'input_value' in JSON payload"}), 400

    try:
        input_value = float(data["input_value"])
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid numeric 'input_value'"}), 400

    # Extract scalar element [0] instead of casting a 1D numpy array directly to float
    prediction = loaded_model.predict([[input_value]])[0]
    return jsonify({"prediction": float(prediction)})


# To run locally or via pyngrok:
# if __name__ == "__main__":
#     from pyngrok import ngrok
#     if os.getenv("NGROK_AUTHTOKEN"):
#         ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])
#     app.run(host="0.0.0.0", port=5000, debug=False)


# ------------------------------------------------------------------------------
# 12. Case Study: House Price Prediction (Leakage-Free & Valid Dataset Source)
# ------------------------------------------------------------------------------
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Valid public mirror of King County House Sales dataset
KC_HOUSE_URL = "https://raw.githubusercontent.com/rashida048/Datasets/master/home_data.csv"
house_df = pd.read_csv(KC_HOUSE_URL)

# Feature Engineering (Strictly excluding target-derived features like price_per_sqft)
reference_year = pd.to_datetime(house_df["date"]).dt.year.max()
house_df["house_age"] = reference_year - house_df["yr_built"]
house_df["renovated"] = (house_df["yr_renovated"] > 0).astype(int)

house_df = pd.get_dummies(house_df, columns=["condition", "grade"], drop_first=True)
house_df = house_df.drop(
    ["id", "date", "yr_built", "yr_renovated", "zipcode", "lat", "long"], axis=1
)

X_house = house_df.drop("price", axis=1)
y_house = house_df["price"]

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

# Note: Tree ensembles don't strictly require scaling, but Pipeline keeps preprocessing clean
house_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "model",
        GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
    ),
])

house_pipeline.fit(X_train_h, y_train_h)
y_pred_h = house_pipeline.predict(X_test_h)

rmse = np.sqrt(mean_squared_error(y_test_h, y_pred_h))
mae = mean_absolute_error(y_test_h, y_pred_h)
r2 = r2_score(y_test_h, y_pred_h)

print("\n=== House Price Prediction Results (Leakage-Free) ===")
print(f"RMSE     : ${rmse:,.2f}")
print(f"MAE      : ${mae:,.2f}")
print(f"R-squared: {r2:.4f}")


# ------------------------------------------------------------------------------
# Practice Problem 1: Customer Segmentation (Index-Aligned K-Means)
# ------------------------------------------------------------------------------
from sklearn.cluster import KMeans


def run_customer_segmentation(excel_path="Online Retail.xlsx", output_path="Customer_Segmentation_Results.xlsx"):
    retail_df = pd.read_excel(excel_path)

    # Filter valid positive transactions and drop missing values BEFORE clustering to preserve alignment
    clean_retail = retail_df.dropna(subset=["Quantity", "UnitPrice"]).copy()
    clean_retail = clean_retail[(clean_retail["Quantity"] > 0) & (clean_retail["UnitPrice"] > 0)]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clean_retail[["Quantity", "UnitPrice"]])

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    clean_retail["Cluster"] = kmeans.fit_predict(scaled_features)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        clean_retail["Quantity"],
        clean_retail["UnitPrice"],
        c=clean_retail["Cluster"],
        cmap="viridis",
        alpha=0.6,
    )
    plt.xlabel("Quantity")
    plt.ylabel("Unit Price")
    plt.title("Customer Segmentation Based on Quantity and Unit Price")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.show()

    clean_retail.to_excel(output_path, index=False)
    print(f"Segmentation results saved to {output_path}")


# ------------------------------------------------------------------------------
# Practice Problem 2: Sentiment Analysis (TF-IDF + Logistic Regression)
# ------------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer


def load_csv_with_encoding(file_path):
    for encoding in ["utf-8", "ISO-8859-1", "latin1", "cp1252"]:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode {file_path} with standard encodings.")


def run_sentiment_analysis(csv_path="test.csv"):
    sent_df = load_csv_with_encoding(csv_path)
    sent_df.columns = [c.strip().lower() for c in sent_df.columns]

    if "text" not in sent_df.columns or "sentiment" not in sent_df.columns:
        raise KeyError("Expected 'text' and 'sentiment' columns in dataset.")

    sent_df = sent_df.dropna(subset=["text", "sentiment"]).copy()
    X_text = sent_df["text"].astype(str)
    y_sent = sent_df["sentiment"].astype(str)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_text, y_sent, test_size=0.2, random_state=42, stratify=y_sent
    )

    sentiment_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])

    sentiment_pipeline.fit(X_tr, y_tr)
    preds = sentiment_pipeline.predict(X_te)
    print("\n=== Sentiment Classification Report ===")
    print(classification_report(y_te, preds))


# ------------------------------------------------------------------------------
# Practice Problem 3: Time Series Analysis (Fixed Column Names: DATE & IPG2211A2N)
# ------------------------------------------------------------------------------
def run_electric_production_ts(csv_path="Electric_Production.csv"):
    ts_df = pd.read_csv(csv_path)

    # Match actual dataset schema ('DATE' and 'IPG2211A2N')
    ts_df["DATE"] = pd.to_datetime(ts_df["DATE"], errors="coerce")
    ts_df = ts_df.dropna(subset=["DATE"]).sort_values("DATE").set_index("DATE")

    plt.figure(figsize=(10, 6))
    plt.plot(ts_df.index, ts_df["IPG2211A2N"], label="Monthly Production", color="tab:blue")
    plt.plot(
        ts_df["IPG2211A2N"].rolling(window=12).mean(),
        label="12-Month Rolling Mean",
        color="tab:orange",
        linewidth=2,
    )
    plt.title("Electric Production Over Time (IPG2211A2N)")
    plt.xlabel("Date")
    plt.ylabel("Production Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# Practice Problem 4: Image Classification (Modern Keras Input Layer & Holdout Test)
# ------------------------------------------------------------------------------
def run_cifar10_cnn(epochs=10):
    from tensorflow.keras import datasets, layers, models

    (X_train_img, y_train_img), (X_test_img, y_test_img) = datasets.cifar10.load_data()
    X_train_img = X_train_img.astype("float32") / 255.0
    X_test_img = X_test_img.astype("float32") / 255.0

    # Use explicit Input(shape=...) layer instead of input_shape inside Conv2D
    cnn_model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    cnn_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Use validation_split on training data so X_test remains a strict holdout set
    history = cnn_model.fit(
        X_train_img,
        y_train_img,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
    )

    test_loss, test_acc = cnn_model.evaluate(X_test_img, y_test_img, verbose=0)
    print(f"\n=== CIFAR-10 Holdout Test Accuracy: {test_acc:.4f} (Loss: {test_loss:.4f}) ===")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("CNN Training vs. Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
