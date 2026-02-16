#
"""
@author: brumartinsg (github)
"""
# %% Bibliotecas 

import pandas as pd
import numpy as np 
import os
print(os.getcwd())

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

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Calcular ROC-AUC
auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", auc)

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()

# %% Importância das variáveis 
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Fatores que Geram Churn')
plt.show()

# %% Validação Matriz de confusão 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()

