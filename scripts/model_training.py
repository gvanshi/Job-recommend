import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load processed data
students_data = pd.read_csv('data/processed/cleaned_students_data.csv')

# Features and Labels (example)
X = students_data[['Feature1', 'Feature2']]  # Replace with your feature columns
y = students_data[['Label1', 'Label2']]  # Replace with your label columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'models/trained_model.pkl')
