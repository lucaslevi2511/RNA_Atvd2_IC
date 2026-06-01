import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

try:
    df = pd.read_csv("Dry_Bean_Dataset.csv")
except FileNotFoundError:
    df = pd.read_excel("Dry_Bean_Dataset.xlsx")
    df.to_csv("Dry_Bean_Dataset.csv", index=False)

X = df.drop("Class", axis=1)
Y = df["Class"]

encoder = LabelEncoder()
y_encoder = encoder.fit_transform(Y)
y_hot = to_categorical(y_encoder)

x_train, x_test, y_train, y_test = train_test_split(
    X, y_hot, train_size=0.3, stratify=y_encoder, random_state=42
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

topologias = {
    "1 Camada (80 neurônios)": [80],
    "1 Camada (90 neurônios)": [90],
    "1 Camada (100 neurônios)": [100],
    "3 Camadas (50 neurônios cada)": [50, 50, 50]
}
momentos = [0.9, 0.7]
LEARNING_RATE = 0.01
EPOCHS = 30  

resultados = []
historicos = {}

for nome_topo, camadas in topologias.items():
    for mom in momentos:
        nome_config = f"{nome_topo} (Momento {mom})"
        print(f"Treinando: {nome_config}...")
        
        model = Sequential()
        model.add(Dense(camadas[0], activation='relu', input_shape=(x_train_scaled.shape[1],)))
        
        for neuronios in camadas[1:]:
            model.add(Dense(neuronios, activation='relu'))
            
        model.add(Dense(7, activation='softmax'))
        
        otimizador = SGD(learning_rate=LEARNING_RATE, momentum=mom)
        model.compile(optimizer=otimizador, loss='categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(
            x_train_scaled, y_train,
            epochs=EPOCHS,
            batch_size=32,
            validation_data=(x_test_scaled, y_test),
            verbose=0
        )
        
        historicos[nome_config] = history.history
        
        train_loss, train_acc = model.evaluate(x_train_scaled, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(x_test_scaled, y_test, verbose=0)
        
        y_test_pred = np.argmax(model.predict(x_test_scaled, verbose=0), axis=1)
        y_test_true = np.argmax(y_test, axis=1)
        
        f1 = f1_score(y_test_true, y_test_pred, average='macro')
        precisao = precision_score(y_test_true, y_test_pred, average='macro', zero_division=0)
        revocacao = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)
        
        resultados.append({
            "Configuração": nome_config,
            "Train Loss": round(train_loss, 4),
            "Test Loss": round(test_loss, 4),
            "Train Acc (%)": round(train_acc * 100, 2),
            "Test Acc (%)": round(test_acc * 100, 2),
            "Precision": round(precisao, 4),
            "Recall": round(revocacao, 4),
            "F1-Score": round(f1, 4)
        })

df_resultados = pd.DataFrame(resultados)

fig_tab, ax_tab = plt.subplots(figsize=(14, 5))
ax_tab.axis('off')
ax_tab.axis('tight')

tabela_visual = ax_tab.table(
    cellText=df_resultados.values, 
    colLabels=df_resultados.columns, 
    loc='center', 
    cellLoc='center',
    colColours=['#40466e']*len(df_resultados.columns) 
)

tabela_visual.auto_set_font_size(False)
tabela_visual.set_fontsize(10)
tabela_visual.scale(1.2, 1.8) 

for (row, col), cell in tabela_visual.get_celld().items():
    if row == 0:
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')

plt.title("Tabela Comparativa de Validação Inicial (Questão 6)", fontsize=14, weight='bold', pad=20)
plt.tight_layout()

fig_curvas, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

for nome_config, hist in historicos.items():
    linha = ax1.plot(hist['loss'], label=f"Treino - {nome_config}")
    ax1.plot(hist['val_loss'], linestyle='--', color=linha[0].get_color(), label=f"Teste - {nome_config}")

ax1.set_title('Curvas de Validação - Função de Perda (Categorical Crossentropy)', fontsize=12, weight='bold')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Loss')
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
ax1.grid(True, linestyle='--', alpha=0.5)

for nome_config, hist in historicos.items():
    linha = ax2.plot(hist['accuracy'], label=f"Treino - {nome_config}")
    ax2.plot(hist['val_accuracy'], linestyle='--', color=linha[0].get_color(), label=f"Teste - {nome_config}")

ax2.set_title('Curvas de Validação - Acurácia', fontsize=12, weight='bold')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Acurácia')
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

plt.show()