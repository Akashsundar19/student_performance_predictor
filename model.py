import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
file = pd.read_csv("student_data.csv")

# Features
X = file[[
    'attendance',
    'study_hours',
    'internal_marks',
    'sleep_hours',
    'assignments_completed'
]]

# Target
Y = file['result']

# Convert labels into numbers
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(Y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
pickle.dump(model, open('student_model.pkl', 'wb'))
pickle.dump(encoder, open('label_encoder.pkl', 'wb'))

print("Model Saved Successfully")                                                    