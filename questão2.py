import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

# ==========================
# Carregar dataset
# ==========================
df = pd.read_excel("Dry_Bean_Dataset.xlsx")

X = df.drop("Class", axis=1)
Y = df["Class"]

# ==========================
# Pré-processamento
# ==========================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(Y)

y_hot = to_categorical(y_encoded)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================
# Hiperparâmetros
# ==========================
learning_rates = [0.001, 0.01, 0.1]
momenta = [0.5, 0.7, 0.9]

# ==========================
# Resultados
# ==========================
results = []

# ==========================
# Função do modelo
# ==========================
def create_model(input_dim, num_classes, lr, momentum):

    optimizer = SGD(
        learning_rate=lr,
        momentum=momentum
    )

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ==========================
# Grid Search
# ==========================
for lr in learning_rates:
    for mom in momenta:

        print(f"\nTreinando -> LR={lr} | Momentum={mom}")

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

        model = create_model(
            X.shape[1],
            y_hot.shape[1],
            lr,
            mom
        )

        history = model.fit(
            X,
            y_hot,
            epochs=200,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )

        # ==========================
        # Encontrar época de convergência
        # ==========================
        losses = history.history['loss']

        convergence_epoch = None

        for i, loss in enumerate(losses):

            if loss <= 0.001:
                convergence_epoch = i + 1
                break

        if convergence_epoch is None:
            convergence_epoch = "Não convergiu"

        final_loss = losses[-1]
        final_acc = history.history['accuracy'][-1]

        # ==========================
        # Salvar resultados
        # ==========================
        results.append({
            "learning_rate": lr,
            "momentum": mom,
            "final_loss": final_loss,
            "final_accuracy": final_acc,
            "epochs_to_converge": convergence_epoch
        })

        # ==========================
        # Curva de convergência
        # ==========================
        plt.plot(
            losses,
            label=f'lr={lr}, m={mom}'
        )

# ==========================
# Gráfico final
# ==========================
plt.title("Curvas de Loss")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ==========================
# Mostrar resultados
# ==========================
results_df = pd.DataFrame(results)

print("\nResultados finais:")
print(results_df)