# AI-ML-Solution-for-Flood-Prediction-in-Nagapatla-Andhra-Pradesh
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset (example: historical flood data)
data = pd.read_csv("nagapatla_flood_data.csv")
X = data[["rainfall_24h", "river_level", "soil_moisture"]]
y = data["flood_occurred"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict flood risk (example input)
risk = model.predict([[120, 8.5, 0.7]])  # High rainfall + rising river
print("Flood Risk:" , "High" if risk[0] else "Low")
