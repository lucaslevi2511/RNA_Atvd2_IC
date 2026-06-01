import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from tensorflow.keras.utils import to_categorical

# ==========================
# Carregar dados
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
# Treino e validação
# ==========================
x_train, x_val, y_train, y_val = train_test_split(
    X,
    y_hot,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ==========================
# Hiperparâmetros fixos
# ==========================
LEARNING_RATE = 0.01
MOMENTUM = 0.9

optimizer = SGD(
    learning_rate=LEARNING_RATE,
    momentum=MOMENTUM
)

# ==========================
# Configurações topológicas
# ==========================
hidden_layers_options = [1, 2, 3]
neurons_options = list(range(10, 101, 10))

# ==========================
# Resultados
# ==========================
results = []

# ==========================
# Função dinâmica da rede
# ==========================
def create_model(
    input_dim,
    num_classes,
    hidden_layers,
    neurons
):

    model = Sequential()

    model.add(Input(shape=(input_dim,)))

    for _ in range(hidden_layers):

        model.add(
            Dense(
                neurons,
                activation='relu'
            )
        )

    model.add(
        Dense(
            num_classes,
            activation='softmax'
        )
    )

    model.compile(
        optimizer=SGD(
            learning_rate=LEARNING_RATE,
            momentum=MOMENTUM
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ==========================
# Experimentos
# ==========================
for hidden_layers in hidden_layers_options:

    for neurons in neurons_options:

        print(
            f"\nTreinando -> "
            f"{hidden_layers} camadas | "
            f"{neurons} neurônios"
        )

        tf.keras.backend.clear_session()

        model = create_model(
            X.shape[1],
            y_hot.shape[1],
            hidden_layers,
            neurons
        )

        start_time = time.time()

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=100,
            batch_size=32,
            verbose=0
        )

        training_time = time.time() - start_time

        # ==========================
        # Métricas finais
        # ==========================
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]

        # ==========================
        # Predições para F1-score
        # ==========================
        predictions = model.predict(x_val, verbose=0)

        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_val, axis=1)

        f1 = f1_score(
            y_true,
            y_pred,
            average='weighted'
        )

        # ==========================
        # Detectar convergência
        # ==========================
        convergence_epoch = "Não convergiu"

        for i, loss in enumerate(history.history['loss']):

            if loss <= 0.001:

                convergence_epoch = i + 1
                break

        # ==========================
        # Salvar resultados
        # ==========================
        results.append({
            "hidden_layers": hidden_layers,
            "neurons": neurons,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "f1_score": f1,
            "training_time": training_time,
            "convergence_epoch": convergence_epoch
        })

        # ==========================
        # Curvas de loss
        # ==========================
        plt.figure(figsize=(8,5))

        plt.plot(
            history.history['loss'],
            label='Train Loss'
        )

        plt.plot(
            history.history['val_loss'],
            label='Validation Loss'
        )

        plt.title(
            f'{hidden_layers} camadas - '
            f'{neurons} neurônios'
        )

        plt.xlabel('Épocas')
        plt.ylabel('Loss')

        plt.legend()

        plt.savefig(
            f'loss_{hidden_layers}layers_{neurons}neurons.png'
        )

        plt.close()

# ==========================
# Tabela final
# ==========================
results_df = pd.DataFrame(results)

print("\nResultados finais:")
print(results_df)

# Salvar CSV
results_df.to_csv(
    "topology_results.csv",
    index=False
)

print("\nResultados salvos.")