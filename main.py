# main.py

import matplotlib.pyplot as plt
# Step 1: Import necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the dataset
data = pd.read_csv('AmesHousing.csv')
print("Dataset loaded successfully!")

# Step 3: Data Cleaning
# Handle missing values
missing_values = data.isnull().sum()
print("Missing values before filling:\n", missing_values[missing_values > 0])
data.fillna(method='ffill', inplace=True)
print("Missing values after filling:\n", data.isnull().sum().sum())

# Check for duplicate entries
duplicates = data.duplicated().sum()
print(f'Duplicates: {duplicates}')
data = data.drop_duplicates()
print("Duplicates removed, if any.")

# Step 4: Exploratory Data Analysis (EDA)
# Distribution of target variable
sns.histplot(data['SalePrice'], kde=True)
plt.title('Distribution of Sale Prices')
plt.show()

# Scatter plot of GrLivArea vs. SalePrice
sns.scatterplot(x=data['GrLivArea'], y=data['SalePrice'])
plt.title('GrLivArea vs. SalePrice')
plt.show()

# Box plot of OverallQual vs. SalePrice
sns.boxplot(x=data['OverallQual'], y=data['SalePrice'])
plt.title('OverallQual vs. SalePrice')
plt.show()

# Correlation heatmap
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 5: Feature Engineering
# Encode categorical variables
le = LabelEncoder()
data['Neighborhood'] = le.fit_transform(data['Neighborhood'])

# Scale numerical features
scaler = StandardScaler()
data[['GrLivArea', 'TotalBsmtSF']] = scaler.fit_transform(data[['GrLivArea', 'TotalBsmtSF']])
print("Feature engineering completed.")

# Step 6: Model Building
# Features and target variable
X = data[['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'Neighborhood']]
y = data['SalePrice']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Train a Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Linear Regression model trained.")

# Train a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest model trained.")

# Step 7: Model Evaluation
# Predictions
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Evaluate Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Evaluate Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Linear Regression MSE: {mse_lr}, R-squared: {r2_lr}')
print(f'Random Forest MSE: {mse_rf}, R-squared: {r2_rf}')

# Step 8: Results Interpretation
# Feature importance from Random Forest
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importance:\n", feature_importance)

# Visualize feature importance
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Feature Importance')
plt.show()

print("Project completed successfully!")
