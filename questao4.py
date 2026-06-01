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
# Separação fixa treino/teste
# ==========================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y_hot,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ==========================
# Percentuais de treino
# ==========================
train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

# ==========================
# Resultados
# ==========================
results = []

# ==========================
# Melhor topologia encontrada
# ==========================
def create_model(input_dim, num_classes):

    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(90, activation='relu'),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=SGD(
            learning_rate=0.01,
            momentum=0.9
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ==========================
# Experimentos
# ==========================
for size in train_sizes:

    print(f"\nTreinando com {int(size*100)}% dos dados")

    # ==========================
    # Subconjunto estratificado
    # ==========================
    if size < 1.0:

        X_train, _, y_train, _ = train_test_split(
            X_train_full,
            y_train_full,
            train_size=size,
            random_state=42,
            stratify=np.argmax(y_train_full, axis=1)
        )

    else:

        X_train = X_train_full
        y_train = y_train_full

    # ==========================
    # Separar validação
    # ==========================
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=np.argmax(y_train, axis=1)
    )

    tf.keras.backend.clear_session()

    model = create_model(
        X.shape[1],
        y_hot.shape[1]
    )

    # ==========================
    # Treinamento
    # ==========================
    start_time = time.time()

    history = model.fit(
        X_subtrain,
        y_subtrain,
        validation_data=(X_val, y_val),
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
    # F1-score validação
    # ==========================
    val_predictions = model.predict(X_val, verbose=0)

    y_pred_val = np.argmax(val_predictions, axis=1)
    y_true_val = np.argmax(y_val, axis=1)

    val_f1 = f1_score(
        y_true_val,
        y_pred_val,
        average='weighted'
    )

    # ==========================
    # F1-score teste
    # ==========================
    test_predictions = model.predict(X_test, verbose=0)

    y_pred_test = np.argmax(test_predictions, axis=1)
    y_true_test = np.argmax(y_test, axis=1)

    test_f1 = f1_score(
        y_true_test,
        y_pred_test,
        average='weighted'
    )

    # ==========================
    # Salvar resultados
    # ==========================
    results.append({
        "train_size": int(size * 100),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "training_time": training_time
    })

# ==========================
# DataFrame final
# ==========================
results_df = pd.DataFrame(results)

print("\nResultados:")
print(results_df)

# ==========================
# Curvas de generalização
# ==========================
plt.figure(figsize=(8,5))

plt.plot(
    results_df["train_size"],
    results_df["val_acc"],
    marker='o',
    label='Validation Accuracy'
)

plt.plot(
    results_df["train_size"],
    results_df["test_f1"],
    marker='o',
    label='Test F1'
)

plt.xlabel("Percentual do treino (%)")
plt.ylabel("Desempenho")

plt.title("Curvas de Generalização")

plt.legend()

plt.savefig("generalization_curves.png")

plt.close()

print("\nGráfico salvo.")