import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle


file = pd.read_csv("student_data.csv")


X = file[[
    'attendance',
    'study_hours',
    'internal_marks',
    'sleep_hours',
    'assignments_completed'
]]


Y = file['result']


encoder = LabelEncoder()
Y = encoder.fit_transform(Y)


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier()
model.fit(X_train, Y_train)


predictions = model.predict(X_test)


accuracy = accuracy_score(Y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
pickle.dump(model, open('student_model.pkl', 'wb'))
pickle.dump(encoder, open('label_encoder.pkl', 'wb'))

print("Model Saved Successfully")                                                    
