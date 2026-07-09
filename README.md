# Customer Churn Prediction App

This project is a Streamlit web application that predicts whether a customer is likely to churn based on banking and customer behavior data. It uses a trained TensorFlow/Keras neural network together with preprocessing artifacts such as a label encoder, one-hot encoder, and feature scaler.

## Project Overview

The goal of this project is to build and deploy a churn prediction model for a bank/customer dataset. The app takes user-entered customer details and returns a churn probability, helping to identify customers who may leave the service.

### Example input features
- Geography
- Gender
- Age
- Balance
- Credit score
- Estimated salary
- Tenure
- Number of products
- Credit card status
- Active membership status

## Data Science Highlights

- Problem type: Binary classification
- Target variable: Customer churn (likely to leave or not)
- Model type: Neural network built with TensorFlow/Keras
- Preprocessing: Encoding of categorical features, scaling of numerical features
- Evaluation strategy: Training and validation metrics were monitored during model fitting

## Model Performance

From the training logs in the project notebook, the model reached approximately:
- Training accuracy: 87.1%
- Validation accuracy: 86.0%
- Training loss: 0.3191
- Validation loss: 0.3529

These results indicate that the model generalizes reasonably well for a churn classification task.

## Features

- Interactive web interface built with Streamlit
- Real-time churn probability prediction
- User-friendly input form for customer attributes
- Pretrained model and preprocessing pipeline included

## Project Files

- app.py: Streamlit application entry point
- model.h5: Trained Keras model
- label_encoder_gender.pkl: Gender label encoder
- onehot_encoder_geo.pkl: Geography one-hot encoder
- scaler.pkl: Feature scaler
- Churn_Modelling.csv: Dataset used for model development
- requirements.txt: Python dependencies

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Run the App

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal in your browser.

## Notes

- The app expects the model and preprocessing files to be present in the project directory.
- The prediction threshold is set to 0.5.
- For a stronger production-ready analysis, additional metrics such as precision, recall, F1-score, and confusion matrix can be added.

## Example

Once the app is running, enter customer details and use the interface to view the churn probability and prediction outcome.
