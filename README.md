# Diabetes Prediction App

A machine learning web app that predicts the likelihood of diabetes based on 
patient health metrics, built with Streamlit.

## Overview

This app takes in key clinical indicators (glucose level, blood pressure, BMI, 
insulin, skin thickness, age, etc.) and uses a trained ML model to predict the 
probability of diabetes. It also displays clinical reference ranges to help 
users interpret their inputs.

## Features

- Interactive input form for health metrics
- Real-time prediction using a trained classification model
- Clinical reference ranges for context
- Clean, easy-to-read summary table of inputs

## Tech Stack

- **Python**
- **Streamlit** — web app framework
- **scikit-learn** — model training and prediction
- **pandas** — data handling

## Dataset

Trained on the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), 
which includes features like glucose, blood pressure, BMI, insulin, and age.

## Installation

1. Clone this repository:

git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction


2. Install dependencies:

pip install -r requirements.txt


3. Run the app:

streamlit run app.py


## Usage

1. Launch the app using the command above.
2. Enter your health metrics in the input fields.
3. Click predict to see your diabetes risk result.
4. Compare your values against the clinical reference ranges shown.

## Disclaimer

This tool is for educational purposes only and is **not a substitute for 
professional medical advice**. Always consult a healthcare provider for 
medical concerns.
