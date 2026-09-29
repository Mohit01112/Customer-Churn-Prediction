# 🏦 Customer Churn Prediction

A Streamlit web application that predicts whether a bank customer is likely to leave (churn) based on customer and account information. The model is built using TensorFlow/Keras and deployed with Streamlit.

## 🚀 Live Demo

https://customer-segmentation-sql-analysis-6q3may9mxwoqmchgctam7x.streamlit.app/

---

## 📌 Project Overview

Customer churn prediction helps banks identify customers who are likely to leave so they can take actions to retain them.

This application allows users to enter customer details and instantly predicts:

- Churn Probability
- Customer Status (Likely to Stay or Leave)

---

## ✨ Features

- User-friendly Streamlit interface
- Real-time churn prediction
- TensorFlow/Keras ANN model
- Data preprocessing using Label Encoding, One-Hot Encoding, and Standard Scaling
- Displays churn probability with prediction result

---

## 📊 Model Performance

| Metric | Value |
|---------|-------|
| Accuracy | **86.70%** |
| Loss | **0.3355** |
| Precision (Churn) | **74%** |
| Recall (Churn) | **50%** |
| F1-Score (Churn) | **59%** |

### Classification Report

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Stay (0) | 0.89 | 0.96 | 0.92 |
| Churn (1) | 0.74 | 0.50 | 0.59 |

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 📂 Project Structure

```
Customer-Churn-Prediction/
│── app.py
│── model.h5
│── scaler.pkl
│── label_encoder_gender.pkl
│── onehot_encoder_geo.pkl
│── Churn_Modelling.csv
│── requirements.txt
│── runtime.txt
│── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Mohit01112/Customer-Churn-Prediction.git
```

Go to the project folder

```bash
cd Customer-Churn-Prediction
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📥 Input Features

- Geography
- Gender
- Age
- Credit Score
- Balance
- Estimated Salary
- Tenure
- Number of Products
- Credit Card Status
- Active Member Status

---

## 📈 Output

The application predicts:

- Churn Probability
- Low Risk (Customer Likely to Stay)
- High Risk (Customer Likely to Churn)

---

## 👨‍💻 Author

**Mohit Jadhav**

- LinkedIn: https://www.linkedin.com/in/mohit-jadhav-49734427b/

---

⭐ If you found this project useful, consider giving it a star!
