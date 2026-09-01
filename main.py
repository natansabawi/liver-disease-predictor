import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Column names for the dataset
columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
           'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
           'Aspartate_Aminotransferase', 'Total_Protiens',
           'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset']

# Load data WITHOUT header row
df = pd.read_csv('data/indian_liver_patient.csv', header=None, names=columns)
print("Data loaded:", df.shape)
print(df.head())

# Fix missing values
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median())

# Convert Male/Female to 1/0
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Features (X) and Target (y)
X = df.drop('Dataset', axis=1)
y = df['Dataset']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numbers
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Done! Model saved.")