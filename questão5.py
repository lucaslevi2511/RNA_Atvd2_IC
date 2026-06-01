import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


try:
    df = pd.read_csv("Dry_Bean_Dataset.csv")
except FileNotFoundError:
    df = pd.read_excel("Dry_Bean_Dataset.xlsx")
    df.to_csv("Dry_Bean_Dataset.csv", index=False)

Y = df["Class"]
X_completo = df.drop("Class", axis=1)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(Y)
y_hot = to_categorical(y_encoded)

# Definição dos Subconjuntos de Atributos
# Analisando a correlação comum neste dataset, criamos 3 cenários:
subconjuntos = {
    "Todas as Features": list(X_completo.columns),
    
    # Removemos Perimeter, ConvexArea e EquivDiameter pois são altamente correlacionados com Area
    "Sem Redundantes": [col for col in X_completo.columns if col not in ['Perimeter', 'ConvexArea', 'EquivDiameter']],
    
    # Focando apenas nos atributos de proporção e forma geométrica
    "Apenas Formato (Shape)": ['AspectRation', 'Eccentricity', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']
}

resultados = []
historicos = {}

EPOCHS = 50
BATCH_SIZE = 32

for nome_exp, features in subconjuntos.items():
    print(f"\n--- Iniciando Experimento: {nome_exp} ({len(features)} atributos) ---")
    
    X_subset = X_completo[features]

    x_temp, x_test, y_temp, y_test = train_test_split(
        X_subset, y_hot, test_size=0.15, stratify=y_encoded, random_state=42
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.1764, stratify=y_temp, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(7, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(
        x_train_scaled, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val_scaled, y_val),
        verbose=0 
    )
    end_time = time.time()
    tempo_treino = end_time - start_time
    
    # Avaliação
    val_loss, val_acc = model.evaluate(x_val_scaled, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(x_test_scaled, y_test, verbose=0)
    
    # Medida-F (F1-Score Macro para múltiplas classes)
    y_val_pred = np.argmax(model.predict(x_val_scaled, verbose=0), axis=1)
    y_val_true = np.argmax(y_val, axis=1)
    f1_val = f1_score(y_val_true, y_val_pred, average='macro')
    
    y_test_pred = np.argmax(model.predict(x_test_scaled, verbose=0), axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    f1_test = f1_score(y_test_true, y_test_pred, average='macro')
    
    # Armazenar resultados
    historicos[nome_exp] = history.history
    resultados.append({
        "Experimento": nome_exp,
        "Qtd Atributos": len(features),
        "Tempo (s)": round(tempo_treino, 2),
        "Val Loss": round(val_loss, 4),
        "Val Acc (%)": round(val_acc * 100, 2),
        "Val F1-Score": round(f1_val, 4),
        "Test F1-Score": round(f1_test, 4)
    })
    print(f"Concluído! Tempo: {tempo_treino:.2f}s | Val Acc: {val_acc*100:.2f}% | Test F1: {f1_test:.4f}")


df_resultados = pd.DataFrame(resultados)

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')

tabela = ax.table(
    cellText=df_resultados.values,
    colLabels=df_resultados.columns,
    cellLoc='center',
    loc='center'
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(10)
tabela.scale(1.2, 1.5)

plt.title("Tabela de Resultados Comparativos", pad=20)
plt.show()

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
for nome_exp, hist in historicos.items():
    plt.plot(hist['val_loss'], label=f"{nome_exp} (Val)")
plt.title('Curvas de Convergência - Função de Perda (Validação)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
for nome_exp, hist in historicos.items():
    plt.plot(hist['val_accuracy'], label=f"{nome_exp} (Val)")
plt.title('Curvas de Convergência - Acurácia (Validação)')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()