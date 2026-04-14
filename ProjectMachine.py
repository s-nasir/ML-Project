import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Loading data sets and then combining into one file
math = pd.read_csv("student-mat.csv", sep=";")
por = pd.read_csv("student-por.csv", sep=";")

df = pd.concat([math, por], ignore_index=True)

# ENFORCING IID ASSUMPTION: Drop duplicate students appearing in both Math and Portuguese sets.
identifying_attributes = ["school","sex","age","address","famsize","Pstatus",
                          "Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]
df = df.drop_duplicates(subset=identifying_attributes, keep='first')


print("Corrected deduplicated dataset shape:", df.shape)

# Making the target variable which is which student will be at risk
df['AtRisk'] = (df['G3'] < 10).astype(int)

# Making the plots

# Class distribution
plt.figure()
sns.countplot(x='AtRisk', data=df)
plt.title("Class Distribution (At-Risk vs Safe)")
plt.savefig("class_distribution.png")

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")

# Absences vs final grade
plt.figure()
sns.scatterplot(x='absences', y='G3', data=df)
plt.title("Absences vs Final Grade")
plt.savefig("absences_vs_grade.png")

# Study time vs final grade
plt.figure()
sns.boxplot(x='studytime', y='G3', data=df)
plt.title("Study Time vs Final Grade")
plt.savefig("studytime_vs_grade.png")

# Adding fatures
df = df.drop(columns=['G3'])

X = df.drop(columns=['AtRisk'])
y = df['AtRisk']

num_cols = X.select_dtypes(include=['int64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Pre processing the dara and splitting it into categories to get rid of invalid data
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Creating models using random forest and logistic regression
# 1. Logistic Regression (High Bias / Low Variance Baseline)
log_model = Pipeline([
    ('preprocess', preprocessor),
    # max_iter=1000 is chosen to ensure the underlying gradient descent 
    # optimization (lbfgs solver) fully converges to the global minimum 
    # of the convex loss surface without throwing ConvergenceWarnings.
    ('model', LogisticRegression(max_iter=1000)) 
])

# 2. Random Forest (Low Bias / High Variance Management)
rf_model = Pipeline([
    ('preprocess', preprocessor),
    # n_estimators=100 is chosen to create sufficient diversity in the ensemble 
    # to reduce variance (via bagging) without incurring unnecessary computational overhead.
    # random_state=42 ensures stochastic reproducibility across runs.
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Validating the data with the calculated scores
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_f1 = cross_val_score(log_model, X, y, cv=skf, scoring='f1')
rf_f1 = cross_val_score(rf_model, X, y, cv=skf, scoring='f1')

print("\n--- Cross Validation ---")
print("Logistic Regression F1:", log_f1.mean())
print("Random Forest F1:", rf_f1.mean())

# Training to improve next set of data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluating the data with the expected value and then the predicted value we came up with
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", roc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("roc_curve.png")

# Figuring out which features matter the most in terms of changing if they are at risk or not
model = rf_model.named_steps['model']
importances = model.feature_importances_

feature_names = rf_model.named_steps['preprocess'].get_feature_names_out()

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.savefig("feature_importance.png")

print("\nTop Features:\n", feat_imp.head(10))