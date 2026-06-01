import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical


df = pd.read_excel("Dry_Bean_Dataset.xlsx")

X = df.drop("Class", axis=1)
Y = df["Class"]

encoder = LabelEncoder()
y_encoder = encoder.fit_transform(Y)
y_hot = to_categorical(y_encoder)

scaler = StandardScaler()
X = scaler.fit_transform(X)

def create_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

seeds = [1, 2, 3, 4, 5]
histories = []

for seed in seeds:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = create_model(X.shape[1], y_hot.shape[1])

    history = model.fit(
        X,
        y_hot,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    histories.append(history)
    print(f"Treinamento com seed {seed} concluído.")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for i, history in enumerate(histories):
    plt.plot(history.history['loss'], label=f'Seed {seeds[i]}')
plt.title('Loss de treinamento')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
for i, history in enumerate(histories):
    plt.plot(history.history['accuracy'], label=f'Seed {seeds[i]}')
plt.title('Accuracy de treinamento')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()