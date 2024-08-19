import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle
import os

# Load data
data = pd.read_csv('../training_data/training.csv')

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode features
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X).toarray()

print(X.head())
print(X.prognosis.describe())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

print(model)
# Save the model
model_save_path = 'model/pocket_doc_model.keras'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)

print(model_save_path)


# Save encoders
encoder_save_path = 'model/encoder.pkl'
with open(encoder_save_path, 'wb') as file:
    pickle.dump(encoder, file)
    print(label_encoder)

label_encoder_save_path = 'model/label_encoder.pkl'
with open(label_encoder_save_path, 'wb') as file:
    pickle.dump(label_encoder, file)



# ######
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
#
# # Load data
# data = pd.read_csv('../training_data/training.csv')
#
# # Separate features and target
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]
#
# # Encode target labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
#
# # One-hot encode features
# encoder = OneHotEncoder()
# X_encoded = encoder.fit_transform(X).toarray()
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
#
# print(X.head())
# print(X.prognosis.describe())
#
#
# #######
#
#
# # Define the model
# model = Sequential()
# model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(label_encoder.classes_), activation='softmax'))
#
# # Compile the model
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
#
# # Save the model
# model.save('/model/pocket_doc_model.h5')
# # model.save('/model/pocket_doc_model.keras')
