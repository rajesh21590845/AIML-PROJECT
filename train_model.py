from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import pandas as pd

# Load dataset
df = pd.read_csv('updated_bengaluru.csv')

# Preprocessing (assuming previous steps are correctly handled)
df['size'] = df['size'].str.extract('(\d+)').astype(float)
df = df.drop(columns=['society', 'availability'])
df = pd.get_dummies(df, columns=['area_type', 'location'], drop_first=True)

# Define features and target
X = df.drop(columns=['price'])
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and feature columns
joblib.dump(model, 'xgb_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
print("Model saved successfully!")
