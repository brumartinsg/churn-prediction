# 📉 Churn Prediction Project

## 🔍 Contexto

Este projeto tem como objetivo **prever o churn de clientes** (cancelamento) utilizando técnicas de *Machine Learning*, com foco não apenas em performance estatística, mas também em **interpretação e aplicação prática para o negócio**.

O problema de churn é crítico em contextos de assinatura e recorrência, pois clientes que cancelam geram impacto direto em receita e crescimento. Antecipar esse comportamento permite ações proativas de retenção.

---

## 🎯 Objetivo

Construir um modelo preditivo capaz de:

* Estimar a probabilidade de churn de cada cliente
* Priorizar clientes com maior risco de cancelamento
* Apoiar decisões de negócio com base em dados

A métrica principal utilizada é **ROC-AUC**, adequada para problemas de classificação binária com classes desbalanceadas.

---

## 🧱 Estrutura do Projeto

```
churn-prediction/
│
├── data/
│   ├── raw/                # Dados brutos
│   └── processed/          # Dados tratados e prontos para modelagem
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│   └── churn_model.pkl
│
├── reports/
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── metrics.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧪 Metodologia

1. **Exploratory Data Analysis (EDA)**

   * Análise de distribuição das variáveis
   * Identificação de padrões associados ao churn

2. **Feature Engineering**

   * Tratamento de variáveis categóricas
   * Criação de variáveis derivadas

3. **Modelagem**

   * Pipeline com pré-processamento + modelo
   * Algoritmos testados: Regressão Logística, Random Forest

4. **Avaliação**

   * ROC-AUC
   * Matriz de confusão
   * Precision e Recall
   * Análise de threshold

---

## 📊 Métricas

* **ROC-AUC** (métrica principal)
* Precision
* Recall
* Confusion Matrix

A escolha da ROC-AUC se dá pela capacidade de avaliar a separação entre churners e não churners independentemente do threshold.

---

## 🔎 Principais Insights

* Clientes com contratos mensais apresentam maior propensão ao churn
* Baixo tempo de permanência (*tenure*) é um forte indicador de risco
* Serviços adicionais de suporte reduzem significativamente a chance de churn

---

## 🛠️ Tecnologias Utilizadas

* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Jupyter Notebook

---

## 🚀 Próximos Passos

* Ajuste fino de hiperparâmetros
* Interpretação do modelo com SHAP
* Simulação de impacto financeiro da retenção
* Deploy do modelo como API

---

## 👤 Autor

Projeto desenvolvido para fins de estudo e portfólio, com foco em **Data Analysis e Machine Learning aplicados a negócio**.
