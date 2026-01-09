import pandas as pd
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Load dataset (FIXED separator)
data = pd.read_csv("dataset/winequality-red.csv")

print("Columns:", list(data.columns))

X = data.drop("quality", axis=1)
y = data["quality"]

# Feature selection
corr = data.corr()["quality"].abs()
selected_features = corr[corr > 0.1].index.drop("quality")
X = X[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "outputs/model.pkl")

# Save results
results = {
    "MSE": mse,
    "R2_Score": r2
}

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=4)
