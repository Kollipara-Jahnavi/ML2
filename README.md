# Machine Learning Assignment 2 - Mobile Price Classification

## 1. Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict the **price range** of a mobile phone based on its technical specifications.

The target variable is **price_range**, which categorizes each mobile phone into one of four classes:
- 0 (Low cost)
- 1 (Medium cost)
- 2 (High cost)
- 3 (Very high cost)

This project also includes a deployed Streamlit web application that allows users to upload a CSV file and evaluate predictions using any selected model.

---

## 2. Dataset Description
Dataset Name: **Mobile Price Classification Dataset**

Dataset Source (Kaggle):
https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

Dataset Properties:
- Total Rows: **2000**
- Total Columns: **21**
- Input Features: **20**
- Target Column: **price_range**
- Problem Type: **Multi-class Classification (4 classes: 0,1,2,3)**

The dataset contains mobile phone specifications such as battery power, RAM, internal memory, pixel resolution, and connectivity features.

---

## 3. Models Implemented
The following machine learning classification models were trained and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest Classifier  
6. XGBoost Classifier  

---

## 4. Evaluation Metrics Used
Each model was evaluated using the following metrics:

- Accuracy  
- AUC (Multi-class One-vs-Rest ROC AUC)  
- Precision (Weighted)  
- Recall (Weighted)  
- F1 Score (Weighted)  
- MCC (Matthews Correlation Coefficient)  

---

## 5. Model Performance Comparison Table
The final performance results are saved in:

`model/metrics.csv`

Below is the comparison table:

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| Decision Tree | 0.8300 | 0.8867 | 0.8319 | 0.8300 | 0.8302 | 0.7738 |
| KNN | 0.5000 | 0.7697 | 0.5211 | 0.5000 | 0.5054 | 0.3350 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Random Forest | 0.8775 | 0.9795 | 0.8776 | 0.8775 | 0.8774 | 0.8368 |
| XGBoost | 0.9350 | 0.9945 | 0.9355 | 0.9350 | 0.9350 | 0.9135 |

---

## 6. Observations Table (Insights)

| Observation | Explanation |
|------------|-------------|
| Logistic Regression achieved the best overall performance | Logistic Regression gave the highest Accuracy (0.9650), AUC (0.9987), and MCC (0.9534), indicating the dataset is highly separable with strong linear patterns. |
| XGBoost was the second-best model | XGBoost performed very well (Accuracy 0.9350, AUC 0.9945) due to its ability to capture nonlinear feature interactions. |
| Random Forest showed strong stable performance | Random Forest achieved good performance (Accuracy 0.8775, AUC 0.9795) and is robust due to ensemble averaging. |
| Decision Tree performed moderately | Decision Tree achieved Accuracy 0.83, but it is more prone to overfitting compared to ensemble methods. |
| Naive Bayes gave good baseline performance | Naive Bayes achieved Accuracy 0.81 and strong AUC (0.9506), showing probabilistic methods can perform well on structured numeric features. |
| KNN performed poorly compared to others | KNN had the lowest Accuracy (0.50) and MCC (0.335), indicating it struggled to separate the classes effectively even after scaling. |
---

## 7. Streamlit Web Application
A Streamlit application is developed to:
- Upload a CSV file (test dataset)
- Select a machine learning model
- View predicted results
- View evaluation results (confusion matrix and classification report)
- View model comparison metrics table

Streamlit App Link:
https://ml2-mobile-price-classifier.streamlit.app/

---

## 8. GitHub Repository Link
GitHub Repository:
https://github.com/Kollipara-Jahnavi/ML2

---

## 9. Project Folder Structure
ML2/
│-- app.py
│-- requirements.txt
│-- README.md
│-- data/
│ └── mobile_price.csv
│-- model/
├── train_models.py
├── metrics.csv
└── saved_models/
├── logistic_regression.pkl
├── decision_tree.pkl
├── knn.pkl
├── naive_bayes.pkl
├── random_forest.pkl
├── xgboost.pkl
└── scaler.pkl


---

## 10. How to Run the Project Locally

### Step 1: Install Dependencies
``` bash
pip install -r requirements.txt

python model/train_models.py

streamlit run app.py
```

## 11. Conclusion

This project successfully implements and compares six machine learning classification models for predicting mobile phone price categories. The best performing model achieved high accuracy and AUC, and the deployed Streamlit application provides an interactive interface for model evaluation and predictions.

Run this command:

``` bash
cat model/metrics.csv
```


