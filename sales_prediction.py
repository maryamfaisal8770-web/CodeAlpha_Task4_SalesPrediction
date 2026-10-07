import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the supplied dataset
df = pd.read_csv("Advertising.csv")

# Remove the unnecessary index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

features = ["TV", "Radio", "Newspaper"]
target = "Sales"

# Clean numeric columns
for column in features + [target]:
    df[column] = pd.to_numeric(df[column], errors="coerce")
df = df.dropna(subset=features + [target])

# Select features and target
X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test-set sales
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")
print("\nModel coefficients:")
for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.3f}")

# Correlations with sales
print("\nCorrelation with Sales:")
print(df[features + [target]].corr()[target].drop(target).sort_values(ascending=False))

# Actual vs predicted plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, alpha=0.8)
mn = min(y_test.min(), predictions.min())
mx = max(y_test.max(), predictions.max())
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.show()
