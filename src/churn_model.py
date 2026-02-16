# -*- coding: utf-8 -*-
"""

@author: brumartinsg (GitHub)
"""
# %% Bibliotecas 

import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

# %% Base de dados 

df = pd.read_csv('customer-churn.csv', sep=',', decimal = '.')

# %% Correção de colunas

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

## Binários
bin_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in bin_cols:
    df[col] = df[col].map({'Yes':1, 'No':0})

## Múltiplas categorias
multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
              'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# %% Distribuição do Churn na base

print(df['Churn'].value_counts(normalize=True))
# 0 (Não) --> 0,73
# 1 (Sim) --> 0,26 
# Base desbalanceada 

# Verificar colunas ainda não numéricas
print(df.dtypes[df.dtypes == 'object'])

# Separar features e target
X = df.drop(['customerID','Churn'], axis=1)
y = df['Churn']

# %% Estrutura do modelo

# Divisão treino/teste (70/30)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

from sklearn.ensemble import RandomForestClassifier

# Criação do modelo
model = RandomForestClassifier(
    n_estimators=200,     # número de árvores
    max_depth=8,          # profundidade máxima
    random_state=42,
    class_weight='balanced'  # importante para lidar com desbalanceamento
)

# Treinamento
model.fit(X_train, y_train)

# Probabilidade de churn
y_prob = model.predict_proba(X_test)[:,1]

# Previsão “binarizada” (0 ou 1) com threshold padrão 0.5
y_pred = model.predict(X_test)

# %% Visualizações

# Calcular ROC-AUC
auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", auc)

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='dodgerblue', lw=3, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.6)

plt.title('Performance do Modelo: Curva ROC', fontsize=16, loc='left', pad=20)
plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
plt.ylabel('Taxa de Verdadeiros Positivos (Recall)', fontsize=12)
plt.legend(loc="lower right", frameon=True)
plt.grid(alpha=0.2)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('curva_roc.png', dpi=300)
plt.show()

# %% Importância das variáveis 

plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=X.columns)

# Gráfico de barras horizontais
importances.nlargest(10).sort_values(ascending=True).plot(kind='barh', color='skyblue', width=0.8)

plt.title('Quais fatores mais influenciam o Churn?', fontsize=16, loc='left', color='#333333', pad=20)
plt.xlabel('Importância Relativa', fontsize=12)
plt.grid(axis='x', alpha=0.3)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('importancia_variaveis.png', dpi=300)
plt.show()

# %% Validação Matriz de confusão 
fig, ax = plt.subplots(figsize=(8, 7))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ficou', 'Churn'])

# Plot customizado
disp.plot(cmap='Blues', ax=ax, colorbar=False)
plt.title('Matriz de Confusão: Realidade vs Previsão', fontsize=16, pad=20)
plt.xlabel('Previsão do Modelo', fontsize=12)
plt.ylabel('Valor Real', fontsize=12)

plt.tight_layout()
plt.savefig('matriz_confusao.png', dpi=300)
plt.show()
