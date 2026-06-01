import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

df = pd.read_excel("Dry_Bean_Dataset.xlsx")

df.to_csv("Dry_Bean_Dataset.csv", index=False)

X = df.drop("Class",axis=1)
Y = df["Class"]

encoder = LabelEncoder()
y_encoder=encoder.fit_transform(Y)

y_hot= to_categorical(y_encoder)

x_train,x_test,y_train,y_test= train_test_split(
    X,
    y_hot,
    train_size=0.3,
    stratify=y_encoder
)

scaler = StandardScaler()

x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)


loss, accuracy = model.evaluate(x_test, y_test)

print(f"Acurácia no teste: {accuracy:.4f}")

predictions = model.predict(x_test)

predicted_class = np.argmax(predictions[0])
real_class = np.argmax(y_test[0])

print("Predição:", encoder.inverse_transform([predicted_class])[0])
print("Classe real:", encoder.inverse_transform([real_class])[0])