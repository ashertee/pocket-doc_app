import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import os

# Example data preparation
import numpy as np
import pandas as pd

# Assuming you have a CSV file with 132 symptom columns and 1 target column
# where each row represents a patient
data = pd.read_csv('../training_data/training.csv')

# Separating features (symptoms) and target (diagnosis)
X = data.iloc[:, :-2]
y = data.iloc[:, 132:-1]

# Check for non-numeric values and convert to numeric if necessary
# Convert X to numeric, coercing errors to NaN
# X = X.apply(pd.to_numeric, errors='coerce')
# y = pd.to_numeric(y, errors='coerce')  # Assuming y is numeric, otherwise skip this

print(X.head())
print(y.head())
print(f"Original dataset size: X={X.shape}, y={y.shape}")

# # Check for any NaN values and handle them
# if X.isnull().values.any() or pd.isnull(y).any():
#     # Option 1: Drop rows with NaN values
#     data.dropna(inplace=True)
#     X = data.iloc[:, :-1]
#     y = data.iloc[:, -1]
#
#     # Option 2: Fill NaN values with a placeholder (e.g., 0)
#     # X = X.fillna(0)
#     # y = y.fillna(0)

# Print dataset shape after handling NaNs
print(f"Dataset size after NaN handling: X={X.shape}, y={y.shape}")



# Convert X and y to NumPy arrays
X = X.values
y = y.values

# Encode the target labels (diagnoses) if necessary
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# # Convert data types explicitly
# X = X.astype('float32')
# y_encoded = y_encoded.astype('int32')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(128, input_dim=132, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Check the size of the dataset before splitting
if X.shape[0] == 0 or y_encoded.shape[0] == 0:
    raise ValueError("The dataset is empty after preprocessing. Please check your data.")



# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred = np.argmax(model.predict(X_test), axis=-1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Save the model for future use
print(model)
print(X_train.shape)
print(y_train.shape)
print(y_pred[:6])
print(y_train[:6])
print(X_test[0])
print(X_test.shape)

# Save the model
model_save_path = 'model/pocket_doc_model.keras'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)

print(model_save_path)
# model.save('symptom_diagnosis_model.h5')


# Save encoders
# encoder_save_path = 'model/encoder.pkl'
# with open(encoder_save_path, 'wb') as file:
#     pickle.dump(encoder, file)
#     print(label_encoder)

label_encoder_save_path = 'model/label_encoder.pkl'
with open(label_encoder_save_path, 'wb') as file:
    pickle.dump(label_encoder, file)


# To load and use the model later:
# from tensorflow.keras.models import load_model
# loaded_model = load_model('symptom_diagnosis_model.h5')
