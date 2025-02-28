import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("weather.csv")

# Display structure
print("Sample Data:")
print(data.head())

# Preprocessing
le = LabelEncoder()
data['Weather'] = le.fit_transform(data['Weather'])  # Encode labels (e.g., Sunny=2, Rainy=1)

# Features & target
X = data[['Temperature', 'Humidity', 'WindSpeed']]
y = data['Weather']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict single input
def predict_weather(temp, humidity, windspeed):
    result = model.predict([[temp, humidity, windspeed]])[0]
    return le.inverse_transform([result])[0]

# Test prediction
print("\nExample Prediction:")
print("Predicted Weather:", predict_weather(25, 70, 10))
