import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# ==================================
# Carregamento
# ==================================

try:
    df = pd.read_csv("Dry_Bean_Dataset.csv")
except:
    df = pd.read_excel("Dry_Bean_Dataset.xlsx")

Y = df["Class"]

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(Y)

# ==================================
# MELHOR CONFIGURAÇÃO DA QUESTÃO 5
# ==================================

features_vencedoras = [
    col for col in df.columns
    if col not in [
        "Class",
        "Perimeter",
        "ConvexArea",
        "EquivDiameter"
    ]
]

X = df[features_vencedoras]

# ==================================
# K-FOLD
# ==================================

K = 5

skf = StratifiedKFold(
    n_splits=K,
    shuffle=True,
    random_state=42
)

# ==================================
# Armazenamento
# ==================================

resultados = []

historicos = []

fold = 1

# ==================================
# LOOP DOS FOLDS
# ==================================

for train_idx, test_idx in skf.split(X, y_encoded):

    print(f"\nFold {fold}/{K}")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)

    model = Sequential([
        Dense(
            64,
            activation='relu',
            input_shape=(X_train.shape[1],)
        ),

        Dense(
            32,
            activation='relu'
        ),

        Dense(
            7,
            activation='softmax'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train_hot,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    historicos.append(history.history)

    loss, acc = model.evaluate(
        X_test,
        y_test_hot,
        verbose=0
    )

    pred = model.predict(
        X_test,
        verbose=0
    )

    pred_classes = np.argmax(pred, axis=1)

    f1 = f1_score(
        y_test,
        pred_classes,
        average='macro'
    )

    resultados.append({
        "Fold": fold,
        "Loss": loss,
        "Accuracy": acc,
        "F1": f1
    })

    fold += 1

# ==================================
# RESULTADOS
# ==================================

df_resultados = pd.DataFrame(resultados)

print("\nResultados por Fold")
print(df_resultados)

# ==================================
# MÉDIAS
# ==================================

print("\nResumo Estatístico")

print(
    f"Loss Média: "
    f"{df_resultados['Loss'].mean():.4f}"
)

print(
    f"Loss Desvio: "
    f"{df_resultados['Loss'].std():.4f}"
)

print(
    f"Accuracy Média: "
    f"{df_resultados['Accuracy'].mean():.4f}"
)

print(
    f"Accuracy Desvio: "
    f"{df_resultados['Accuracy'].std():.4f}"
)

print(
    f"F1 Média: "
    f"{df_resultados['F1'].mean():.4f}"
)

print(
    f"F1 Desvio: "
    f"{df_resultados['F1'].std():.4f}"
)

# ==================================
# TABELA
# ==================================

fig, ax = plt.subplots(figsize=(8,3))

ax.axis("off")

tabela = ax.table(
    cellText=df_resultados.round(4).values,
    colLabels=df_resultados.columns,
    loc="center",
    cellLoc="center"
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(10)
tabela.scale(1.2, 1.5)

plt.title("Resultados da Validação Cruzada")
plt.show()

# ==================================
# CURVAS DE LOSS
# ==================================

plt.figure(figsize=(10,5))

for i, hist in enumerate(historicos):

    plt.plot(
        hist["val_loss"],
        label=f"Fold {i+1}"
    )

plt.title("Loss de Validação por Fold")

plt.xlabel("Épocas")
plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.show()

# ==================================
# CURVAS DE ACCURACY
# ==================================

plt.figure(figsize=(10,5))

for i, hist in enumerate(historicos):

    plt.plot(
        hist["val_accuracy"],
        label=f"Fold {i+1}"
    )

plt.title("Accuracy de Validação por Fold")

plt.xlabel("Épocas")
plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.show()