import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load IIOT sensor data (ensure 'temperature', 'torque', 'vibration', 'failure' columns)
df = pd.read_csv("iiot_sensor_data.csv")

# Features and target
X = df[['temperature', 'torque', 'vibration']]
y = df['failure']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
