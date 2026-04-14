# Predicting Student Academic Performance Using Machine Learning

**Group 22 Project Team Members:** Best Akinlabi, Ethan McLeod, George Kassar, Syed Nasir Hussain Naqvi

## Project Overview
This project applies machine learning methodologies to forecast student success using the UCI Student Performance Dataset. The goal is to proactively identify students at risk of academic failure (G3<10) versus those who are "Safe" (G3>=10). By accurately identifying "At-Risk" students, educational institutions can implement timely intervention strategies.

## Dataset
**Source:** UCI Student Performance Dataset (Mathematics and Portuguese courses).

**Preprocessing:** The datasets are combined, and strict deduplication protocols are applied to ensure the Independent and Identically Distributed (IID) assumption, resulting in 649 unique student records.

## Models Used
- **Logistic Regression:** High bias / low variance baseline model.
- **Random Forest:** Low bias / high variance ensemble model (Chosen Model).

## Dependencies
Since there is no requirements.txt, please ensure you have the following Python libraries installed before running the project:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install them via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run
1. Ensure the dataset files (student-mat.csv and student-por.csv) are located in the same directory as the script.
2. Execute the main Python script:

```bash
python ProjectMachine.py
```

The script will output the cross-validation scores, classification report, and ROC-AUC score to the terminal. It will also generate and save several visualizations (e.g., confusion_matrix.png, roc_curve.png, feature_importance.png) directly into the working directory.

## Key Results
**Performance:** The Random Forest model achieved an Accuracy of 88%, an F1-Score of 0.78 for the At-Risk class, and an ROC-AUC of 0.969 on the test set.

**Feature Importance:** Prior academic performance (G1 and G2 grades) were identified as the most dominant predictive features, followed by past failures and absences.
