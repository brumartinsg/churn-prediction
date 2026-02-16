# 📉 Predição de Churn
Autor: Bruna Martins

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

Este projeto aplica Machine Learning para prever o cancelamento de clientes (Churn). Através de uma análise de dados históricos, o modelo identifica padrões que indicam a probabilidade de um cliente deixar a empresa, permitindo ações proativas de retenção.

## 📋 Resumo do Projeto
O Churn é uma das métricas mais críticas para empresas de serviços. Neste projeto:
- Tratamos dados ausentes e convertemos variáveis categóricas via **One-Hot Encoding**.
- Lidamos com o **desbalanceamento de classes** (apenas 26% de churn na base) utilizando pesos balanceados no algoritmo.
- Utilizamos o **Random Forest Classifier** pela sua robustez e facilidade de interpretação através da importância das variáveis.

## 🧠 Premissas e Tratamento de Dados
Para garantir a integridade do modelo, foram adotadas as seguintes premissas:
- **Conversão de Dados:** A coluna `TotalCharges` continha valores vazios (strings com espaços) que impediam a análise numérica. Eles foram convertidos para `NaN` e as 11 linhas afetadas foram removidas, pois representavam menos de 0,15% da base.
- **Engenharia de Variáveis:** - Variáveis categóricas foram transformadas em numéricas (Mapeamento Direto para binárias e One-Hot Encoding para múltiplas categorias).
  - A coluna `customerID` foi removida da modelagem por ser apenas um identificador único sem poder preditivo.
- **Desbalanceamento:** A base apresentava ~73% de retenção e ~27% de churn. Para evitar um modelo tendencioso, utilizamos o parâmetro `class_weight='balanced'` no Random Forest.

## 📊 Resultados Alcançados
- **ROC-AUC: 0.84** (Excelente capacidade de distinção entre classes).
- **Recall de 76%**: O modelo consegue identificar a grande maioria dos clientes que realmente pretendem sair.

### Top 10 Preditores de Churn
Conforme o gráfico de `feature_importances_`, os fatores que mais influenciam a decisão do cliente são:
1. **Tenure** (Tempo de contrato)
2. **TotalCharges** e **MonthlyCharges** (Fatores financeiros)
3. **Tipo de Contrato** (Mensal vs Bienal)

## 📁 Estrutura do Repositório
- `churn_analysis.py`: Script principal com o pipeline de dados e modelo.
- `customer-churn.csv`: Base de dados utilizada.
- `images/`: Gráficos gerados (Curva ROC, Matriz de Confusão, Importância).

---


