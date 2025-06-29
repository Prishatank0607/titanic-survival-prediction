# Titanic Survival Prediction

This project presents a machine learning-based approach to predict passenger survival on the Titanic. It involves training multiple classification models and evaluating their performance using metrics such as accuracy, precision, recall, and F1-score. To enhance model performance, hyperparameter tuning is applied using both GridSearchCV and RandomizedSearchCV. The final step involves analyzing the evaluation results to identify and select the best-performing model for the classification task.

---

## About the Project

The goal of this project is to apply supervised machine learning algorithms to predict whether a passenger survived the Titanic disaster, based on attributes like age, sex, fare, class, and embarkation point. The project implements data preprocessing, model training, performance evaluation, and hyperparameter tuning using both GridSearchCV and RandomizedSearchCV.

---

## Dataset Overview

The dataset used is from Kaggle's **[Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)** competition and includes the following files:

- `train.csv`: Main dataset with labeled data for training
- `test.csv`: Unlabeled data for prediction (optional in this project)
- `gender_submission.csv`: Sample submission file for Kaggle

---

## Features Implemented

- Data preprocessing:
  - Handling missing values
  - Label encoding and scaling
- Trained models:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest (Default & Tuned)
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Hyperparameter Tuning:
  - GridSearchCV
  - RandomizedSearchCV
- Cross-validation with F1-score
- Confusion matrices and ROC curves
- Final model selection with performance analysis

---

## Tools & Technologies Used

- Programming Language: Python
- Notebook Environment: Jupyter Notebook
- Libraries:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn` (for models, metrics, preprocessing, and tuning)
  - `joblib` (for optional model saving)

---

## How to Run

Follow the steps below to set up and run the Jupyter Notebook:

```bash
# Clone the repository
git clone https://github.com/Prishatank0607/titanic_survival_prediction.git

# Navigate to the project directory
cd titanic_survival_prediction

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # For macOS/Linux
# venv\Scripts\activate       # For Windows (use this instead)

# Install required dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook

# Open and run the notebook
titanic_survival_prediction.ipynb
