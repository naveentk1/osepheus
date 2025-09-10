import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the cleaned CSV
df = pd.read_csv("nse_index_features.csv")

# Drop non-predictive columns
df = df.drop(columns=['RequestNumber', 'Index Name', 'INDEX_NAME', 'Index'], errors='ignore')

# Convert date to datetime
df['HistoricalDate'] = pd.to_datetime(df['HistoricalDate'], dayfirst=True)

# Sort by date
df = df.sort_values('HistoricalDate')

# Feature engineering
df['Return'] = df['CLOSE'].pct_change()
df['MA5'] = df['CLOSE'].rolling(window=5).mean()
df['MA10'] = df['CLOSE'].rolling(window=10).mean()
df['MA20'] = df['CLOSE'].rolling(window=20).mean()
df['Vol5'] = df['Return'].rolling(window=5).std()
df['Vol10'] = df['Return'].rolling(window=10).std()
df['Vol20'] = df['Return'].rolling(window=20).std()

# Drop rows with NaN values after feature engineering
df = df.dropna()

# Create target: 1 if next day close is higher, else 0
df['Target'] = (df['CLOSE'].shift(-1) > df['CLOSE']).astype(int)

# Prepare features and target
feature_cols = ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'MA5', 'MA10', 'MA20', 'Vol5', 'Vol10', 'Vol20']
X = df[feature_cols]
y = df['Target'][:-1]  # Last row has no target
X = X[:-1]  # Align X with y

# Time-based train-test split (80% train, 20% test)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Save the model
joblib.dump(model, "nse_index_rf_model.pkl")
print("Model saved as nse_index_rf_model.pkl")
