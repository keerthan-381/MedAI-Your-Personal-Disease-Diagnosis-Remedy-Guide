import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical




train_data = pd.read_csv("Training.csv")
test_data = pd.read_csv("Testing.csv")

train_data = train_data.iloc[:, :-1]
# test_data = test_data.iloc[:, :-1]

# print(test_data.head())

X_train = train_data.drop("prognosis", axis=1)
y_train = train_data["prognosis"]

X_test = test_data.drop("prognosis", axis=1)
y_test = test_data["prognosis"]

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    Conv1D(32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train_onehot.shape[1], activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_onehot, epochs = 10, batch_size = 32, validation_split = 0.2)

y_pred_onehot = model.predict(X_test)
y_pred = np.argmax(y_pred_onehot, axis=1)

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(accuracy_score(y_test, y_pred))

# Save the trained model
model.save("cnn_prognosis_model.h5")
print("Model saved as cnn_prognosis_model.h5")


