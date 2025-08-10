# 🍷 Wine Quality Prediction using Machine Learning

A Machine Learning project that predicts whether wine is **good** or **bad** based on its physicochemical properties.  
Built using a **Random Forest Classifier** and the UCI Red Wine Quality dataset.

---

## 📌 Problem Statement

Wine quality assessment is often done by human tasters, which can be subjective and time-consuming.  
This project automates the process using machine learning to classify wine into **good quality** or **bad quality** based on chemical analysis.

---

## 📂 Dataset Overview

- **Source**: [UCI Red Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Size**: 1,599 samples × 12 features
- **Features**:
  - Fixed Acidity
  - Volatile Acidity
  - Citric Acid
  - Residual Sugar
  - Chlorides
  - Free Sulfur Dioxide
  - Total Sulfur Dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Target**: Quality (`1` = Good Quality, `0` = Bad Quality) based on a threshold score of ≥7

---

## ⚙️ Tech Stack

- **Python**
- **NumPy**, **Pandas**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
  - RandomForestClassifier
  - Train-Test Split
  - Accuracy Score

---

## 🚀 Workflow

1. **Data Loading**
   - Load CSV dataset into Pandas DataFrame
2. **Data Exploration**
   - Statistical summary
   - Missing value check
   - Visualization of quality distribution & feature relationships
3. **Feature Engineering**
   - Binarize quality label: `1` (good) if ≥7, else `0`
4. **Model Training**
   - Random Forest Classifier
5. **Evaluation**
   - Accuracy on training & test sets
6. **Prediction System**
   - Accepts new wine parameters and predicts quality

---

## 📊 Model Performance

| Dataset       | Accuracy |
|---------------|----------|
| Training Set  | ~100%    |
| Test Set      | ~92%     |

---

## 💡 Sample Prediction

```python
input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_data_as_numpy_array)

if prediction[0] == 1:
    print("Good quality wine")
else:
    print("Bad quality wine")
