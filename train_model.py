import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# California Housing dataset
data = fetch_california_housing()

# Input features
X = data.data

# Target value
y = data.target

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)
# Save trained model
obj = {
    "lr": model,
    "columns": data.feature_names
}

joblib.dump(obj, "california.joblib")

print("Model saved successfully!")