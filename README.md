# 📧 Spam Email Classifier

## 📌 Overview
This project implements a **Spam Email Classification system** using TF-IDF vectorization and Logistic Regression.  
It classifies email messages as Spam or Ham (Not Spam).

---

## 📂 Dataset

Dataset used: SMS Spam Collection Dataset

Download from:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Place `mail_data.csv` inside the project directory before running.

Note: Dataset is not included in this repository.

---

## 🛠 Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

---

## 🔍 Methodology

1. Data preprocessing and null handling  
2. Label encoding (spam = 0, ham = 1)  
3. Train-test split (80-20)  
4. TF-IDF feature extraction  
5. Logistic Regression training  
6. Model evaluation using:
   - Training Accuracy
   - Test Accuracy
   - Confusion Matrix
   - Classification Report  

---

## 📊 Model Performance

```text
Training Accuracy: ~0.98
Test Accuracy: ~0.96
```

---

## 🚀 Example Prediction

```text
Input:
"Congratulations! You have won a free lottery ticket."

Output:
Spam
```

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place dataset file inside project folder.

3. Run:

```bash
python spam_classifier.py
```

---

## 📁 Project Structure

```text
spam-email-classifier/
│
├── spam_classifier.py
├── requirements.txt
└── README.md
```

