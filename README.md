# 📧 Spam Email Classifier

## 📌 Overview
This project implements a **Spam Email Classification system** using TF-IDF vectorization and Logistic Regression.  
It classifies email messages as **Spam** or **Ham (Not Spam)** based on text features.

---

## 🛠 Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

---

## 🔍 Methodology

1. Data preprocessing and null value handling  
2. Label encoding (spam = 0, ham = 1)  
3. Train-test split (80-20)  
4. TF-IDF feature extraction  
5. Logistic Regression model training  
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

(Exact values may vary slightly depending on split)
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

1. Download the dataset (`mail_data.csv`)  
2. Place it inside the project directory  

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run:

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
