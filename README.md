# Titanic Survival Prediction

This project is a machine learning solution to predict passenger survival from the Titanic dataset. It is designed to train multiple classification models, evaluate their performance using key metrics, optimize hyperparameters, and select the best-performing model.

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

## How to Run the Project

# 1. Clone the repository
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction

# 2. (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt

# If requirements.txt is not present, install manually:
pip install pandas numpy matplotlib seaborn scikit-learn joblib

# 4. Launch Jupyter Notebook
jupyter notebook
Navigate to titanic_survival_prediction.ipynb
