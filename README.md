# DS-471 Machine Learning Project — Credit Card Default Prediction

Predicting which credit-card clients will **default on their next payment**, comparing four traditional machine-learning models with a neural network.

- **Dataset:** [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Yeh, 2009) — 30,000 clients of a bank in Taiwan, 23 features, 22.1% defaults
- **Models:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (`HistGradientBoostingClassifier`), Neural Network (multilayer perceptron)
- **Course:** DS-471 Machine Learning

| Group member | Student ID |
|---|---|
| Thamer Bander Almutairi | S230006300 |
| Abdullah Mutlaq Albugami | S220014331 |
| Khalid Alhamalawi | S220016917 |
| Khalid Alqarni | S220022107 |
| Mashaal M Alghamdi | S220012786 |

## Results

CV = 5-fold cross-validation on the training set. The test set (4,495 clients, 995 defaults) was used once. The F1, recall and precision columns use the decision threshold tuned on the validation set.

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Recall | Precision | Brier |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.783 | 0.768 | 0.535 | 0.522 | 0.512 | 0.534 | 0.186 |
| Decision Tree | 0.778 | 0.763 | 0.534 | 0.529 | 0.556 | 0.504 | 0.187 |
| Random Forest | 0.791 | 0.776 | 0.559 | 0.527 | 0.508 | 0.547 | 0.172 |
| **Gradient Boosting** | 0.791 | 0.777 | 0.567 | 0.541 | 0.564 | 0.520 | 0.135 |
| Neural Network | 0.784 | 0.772 | 0.550 | 0.521 | 0.510 | 0.533 | 0.136 |

**Recommended model: Gradient Boosting.** It has the best test ROC-AUC, PR-AUC and F1, and the best-calibrated probabilities. With a threshold of 0.29, it catches 56% of the defaulters with 52% precision.

The neural network (128-64-32 neurons, L2 alpha = 1, early stopping at epoch 28) came third. It beat logistic regression and the decision tree on PR-AUC.

## Workflow (notebook sections)

1. **Data preparation and EDA**
   - removed 35 duplicate rows;
   - merged undocumented category codes;
   - interpreted the repayment-status codes;
   - kept and transformed the outliers (signed log / Yeo-Johnson);
   - one-hot encoding;
   - 9 engineered features (utilisation, repayment ratio, payment delays);
   - correlation analysis and feature relevance.
2. **Data splitting:** 70 / 15 / 15, stratified.
3. **Model development:** pipelines for all five models, baseline cross-validation, and an ablation of the feature sets and scalings.
4. **Hyperparameter optimisation:**
   - grid search with 5-fold CV;
   - validation and learning curves;
   - neural-network training/validation curves with early stopping;
   - under- and over-fitting experiments and a seed-robustness check;
   - decision thresholds tuned on the validation set.
5. **Final evaluation:**
   - test metrics, confusion matrices, ROC and precision-recall curves;
   - bootstrap confidence intervals and paired comparisons;
   - calibration and permutation importance;
   - final comparison.

## How to run

Open the notebook in Google Colab:
`https://colab.research.google.com/github/Alhelfi1/DS471-Credit-Card-Default-Prediction/blob/main/DS471_Credit_Card_Default_Prediction.ipynb`

Or run it locally:

```bash
pip install -r requirements.txt
jupyter notebook DS471_Credit_Card_Default_Prediction.ipynb
```

Run all cells; it takes about 25 minutes on a 2-core CPU. The data file is in `data/`. If it is missing, the notebook downloads the original file from the UCI repository. A fixed seed (42) makes the results reproducible.

## Repository structure

| Path | Content |
|---|---|
| `DS471_Credit_Card_Default_Prediction.ipynb` | Complete, executed notebook (Stages 1–5, discussion, conclusion) |
| `data/default_of_credit_card_clients.csv` | Dataset (exact CSV copy of the UCI file `default of credit card clients.xls`) |
| `report/DS471_Project_Report.docx` | Project report |
| `figures/` | All 17 figures (PNG, 200 dpi) |
| `results/` | All results tables (CSV/JSON) |
| `requirements.txt` | Python dependencies |

## Data source and licence

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H — licensed under CC BY 4.0.
