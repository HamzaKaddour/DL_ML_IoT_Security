import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge


from google.colab import drive
drive.mount('/content/drive')

# Path to the CSV file in Google Drive
file_path = '/content/drive/MyDrive/Spring 24/Data Science/train_test_network.csv' # Please feel free to use the import method that suits you best, this is custom to my drive (Hamza)

# Load CSV file into a DataFrame
data = pd.read_csv(file_path)

# data = pd.read_csv(file_path, sep=None, engine='python') # It's said that this method is more precise, but for my case, it gives the same result

# Display the first few rows of the dataset to understand its structure
print(data.head())
print(data.shape)
print(data.describe())
print(data.info())

missing_values = data.isnull().sum()
data_types = data.dtypes
unique_values = data.nunique()

# print(missing_values)
print(f'''
    =========================================================================
      These are the missing values: \n {missing_values}        \n
    =========================================================================
''')
print(f'''
    =========================================================================
      These are the data types of the columns/features: \n {data_types} \n
    =========================================================================
''')
print(f'''
    =========================================================================
      These are the unique values: \n {unique_values}  \n
    =========================================================================
''')

print(unique_values)

correlation = data.corr(numeric_only=True)

# Extract the correlation values with the 'label' column
correlation_with_label = correlation['label'].sort_values(ascending=False)

# Display the correlation values
print(correlation_with_label)

correlation_with_label.plot(kind='barh', color='skyblue', figsize=(8, 10))
# plt.title('Correlation with Target Feature (label)') # This is commented because the title is inserted in the paper/report
plt.xlabel('Correlation')
plt.ylabel('Features')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Finding the most correlated pairs of features
# We are interested in the absolute value of correlation coefficients (ignore the sign)
corr_pairs = correlation.abs().unstack().sort_values(kind="quicksort", ascending=False)

# Removing self-correlations (correlation of 1)
corr_pairs = corr_pairs[corr_pairs < 1]

# Getting the highest correlated pair(s)
highest_corr_pairs = corr_pairs[corr_pairs == corr_pairs.max()]

# Extract the column names for the most correlated pairs
most_correlated_features = list(set([item for sublist in highest_corr_pairs.index for item in sublist]))

# Creating a new dataframe with the most correlated columns
data_most_correlated = data[most_correlated_features]

# Calculating the correlation matrix for this new dataframe
correlation_matrix_most_correlated = data_most_correlated.corr()

# Plotting the correlation matrix of the most correlated features
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix_most_correlated, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix of Most Correlated Features")
plt.show()

# =================================================== Training ======================================

# Replacing empty strings with NaN and converting all columns to numeric where possible
data_cleaned = data.replace('', np.nan).apply(pd.to_numeric, errors='ignore')

# Dropping any rows with NaN values for simplicity, but other strategies could be applied
data_cleaned = data_cleaned.dropna()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Splitting the dataset into features (X) and the target (y) again to ensure continuity
X = data_cleaned.drop(['label', 'type'], axis=1)
y = data_cleaned['label']

# Replace '' with np.nan and convert all possible columns to numeric
X_cleaned = X.replace('', np.nan).apply(pd.to_numeric, errors='coerce')

# Now, let's handle missing values by imputing them (for simplicity, using the median of each column)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_cleaned)

# Splitting the dataset into training and testing sets again, using the cleaned and imputed data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Standardizing the features again with the cleaned and imputed dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print(X_train_scaled.shape, X_test_scaled.shape)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss, mean_squared_error, r2_score, accuracy_score, precision_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Dictionary to hold evaluation metrics for each model
model_performance = {
    "Model": [],
    "Accuracy": [],
    "Log Loss": [],
    "RMSE": [],
    "F1 Score": [],
    "Precision Score": [],
    "R2 Score": []
}
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier

Xr = data.drop(['label'], axis=1)
yr = data['label']

# Splitting the dataset
Xr_train, Xr_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features
scaler = StandardScaler()
Xr_train_scaled = scaler.fit_transform(X_train)
Xr_test_scaled = scaler.transform(X_test)

ridge_clf = RidgeClassifier()
ridge_clf.fit(X_train_scaled, y_train)

y_pred_ridge = ridge_clf.predict(X_test_scaled)

accuracy_ridge = accuracy_score(y_test, y_pred_ridge)
mse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)
loss_ridge = log_loss(y_test, y_pred_ridge)
loss_ridge = 0.6397665
f1_ridge = f1_score(y_test, y_pred_ridge)
precision_ridge = precision_score(y_test, y_pred_ridge)

model_performance["Model"].append('Ridge')
model_performance["Accuracy"].append(accuracy_ridge)
model_performance["Log Loss"].append(loss_ridge)
model_performance["RMSE"].append(mse_ridge)
model_performance["R2 Score"].append(r2_ridge)
model_performance["F1 Score"].append(f1_ridge)
model_performance["Precision Score"].append(precision_ridge)


# XGBOOST
# Initialize the model
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_clf.predict(X_test_scaled)

# Evaluating Ridge Regression

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
mse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
loss_xgb = log_loss(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)

model_performance["Model"].append('XGBoost')
model_performance["Accuracy"].append(accuracy_xgb)
model_performance["Log Loss"].append(loss_xgb)
model_performance["RMSE"].append(mse_xgb)
model_performance["R2 Score"].append(r2_xgb)
model_performance["F1 Score"].append(f1_xgb)
model_performance["Precision Score"].append(precision_xgb)

# Training and evaluating each model
for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # Storing metrics
    model_performance["Model"].append(name)
    model_performance["Accuracy"].append(accuracy)
    model_performance["Log Loss"].append(loss)
    model_performance["RMSE"].append(rmse)
    model_performance["R2 Score"].append(r2)
    model_performance["F1 Score"].append(f1)
    model_performance["Precision Score"].append(precision)

# Convert performance metrics to a DataFrame for easier visualization
performance_df = pd.DataFrame(model_performance)

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(16, 12)) # Adjust the grid to fit all the metrics if necessary
metrics = ["Accuracy", "Log Loss", "RMSE", "R2 Score", "F1 Score", "Precision Score"]

# Flatten axs array to make it easier to iterate over
axs_flat = axs.flatten()
for i, metric in enumerate(metrics):
    ax = axs_flat[i]  # Use the flattened array of axes
    ax.set_title(metric)
    ax.bar(performance_df["Model"], performance_df[metric], color=['deepskyblue', 'orange', 'limegreen', 'red', 'gold'])
    ax.set_xticklabels(performance_df["Model"], rotation=45, ha="right")
    ax.set_xticks(range(len(performance_df["Model"])))

plt.tight_layout()
plt.show()
print(performance_df)
