# Sepsis Survival Prediction using Tabular Machine Learning

## 1. Problem Statement

Sepsis is a critical medical condition where early identification of high-risk patients can significantly improve outcomes. This project aims to predict patient survival using clinical tabular data, with a focus on handling class imbalance and improving detection of high-risk cases.

---

## 2. Approach

The dataset was preprocessed to handle missing values and categorical variables. Multiple models were trained and evaluated to compare performance across different levels of model complexity:

* Logistic Regression (baseline)
* XGBoost (tree-based ensemble)
* TabNet (deep learning model for tabular data)

Model evaluation was performed using ROC-AUC, PR-AUC, F1-score, and recall, with particular emphasis on recall for the minority class (patients who did not survive).

---

## 3. Results

All models achieved similar performance on validation data (ROC-AUC ~0.69). However, a noticeable drop in performance was observed on the test set, indicating potential distribution shift or limited feature signal.

| Model               | ROC-AUC (Test) | PR-AUC (Test) | Recall             |
| ------------------- | -------------- | ------------- | ------------------ |
| Logistic Regression | ~0.55          | ~0.85         | Moderate           |
| XGBoost             | ~0.57          | ~0.87         | Balanced           |
| TabNet              | ~0.57          | ~0.87         | Similar to XGBoost |

TabNet did not provide measurable improvement over tree-based methods while incurring significantly higher computational cost.

---

## 4. Key Observations

* Model performance was consistent across architectures, suggesting that predictive signal is limited by feature quality rather than model capacity
* Performance degradation from validation to test data indicates potential distribution shift
* Default classification thresholds favored the majority class and reduced detection of high-risk patients
* Adjusting the decision threshold significantly improved recall for the minority class

---

## 5. Key Takeaway

In imbalanced clinical datasets, optimizing decision thresholds and evaluation metrics aligned with the problem objective has a greater impact than increasing model complexity.

---

## 6. Deployment Considerations

In a real-world setting, the model can be used as a screening tool with a lower decision threshold to prioritize early detection of high-risk patients. This approach accepts a higher false positive rate in exchange for improved recall, which is appropriate in high-risk clinical scenarios.

---

## 7. Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* PyTorch TabNet
* Matplotlib, Seaborn

---

## 8. Project Structure

notebooks/
  sepsis_survival.ipynb

---

## 9. Future Work

* Feature engineering to improve predictive signal
* Cross-validation to improve generalization
* Model calibration for better probability estimates
