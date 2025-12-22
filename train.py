import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load Dataset

df = pd.read_csv("car data.csv")

# Data Cleaning

df.dropna(inplace=True)

if 'Car_Name' in df.columns:
    df.drop('Car_Name', axis=1, inplace=True)

# Encoding Categorical Columns

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Feature & Target Split

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.values)
X_test = scaler.transform(X_test.values)

# Models

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor()
}

# Training & Evaluation

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n{name}")
    print("R2 Score:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)

    if r2 > best_score:
        best_score = r2
        best_model = model

# Save Best Model & Scaler

joblib.dump(best_model, "best_car_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Best model saved successfully")

