[Uploading README.md…]()
# Medical ML Model Generator - Synthetic Data Simulator

A Streamlit-based web application for generating synthetic medical data, training multiple machine learning classification models, and evaluating their performance in predicting patient health risks.

This project is submitted as the final requirement for CSEC 413 - Modeling and Simulation.

---

## Project Overview

This application simulates a medical health risk screening system. It generates a synthetic patient dataset with known clinical properties, preprocesses the data, trains 10 different machine learning algorithms (including Linear, Tree-based, and Ensemble models), and evaluates their performance using comprehensive visualizations and metrics tables.

---

## Key Features

- Synthetic Data Generation: Automatically generates realistic patient records (Age, BMI, Blood Pressure, Cholesterol, etc.) with customizable sample sizes.
- Multiple Models: Trains 10 classification algorithms simultaneously:
  - LogisticRegression, RidgeClassifier
  - RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
  - SVC, LinearSVC
  - GaussianNB, KNeighborsClassifier
  - MLPClassifier (Neural Network)
- Cross-Validation: Supports Stratified K-Fold Cross-Validation to ensure model stability.
- Advanced Visualizations: 
  - 3D Feature Visualization (Plotly)
  - Learning Curves & Confusion Matrices
  - ROC Curves & Precision-Recall Curves
  - Model Comparison Radar Chart
- Downloadable Results: Users can download trained models and scalers as .pkl files, as well as the original and scaled datasets as .csv files.
- Modern UI: Clean, eye-friendly sidebar and uniform boxed layout with gradient metrics cards.

---

## Tech Stack

- Language: Python 3.8+
- Web Framework: Streamlit
- Machine Learning: Scikit-learn
- Data Manipulation: Pandas, NumPy
- Visualization: Plotly, Matplotlib, Seaborn
- Imbalanced Data Handling: Imbalanced-learn (SMOTE)

---

## Installation & Setup

Follow these steps to run the application locally on your machine.

Prerequisites
Ensure you have Python installed on your computer. You can download it from python.org.

Step 1: Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/your-username/medical-ml-simulator.git
cd medical-ml-simulator
