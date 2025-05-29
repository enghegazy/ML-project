# Data Cleaning and Modeling Project

## Project Goal
The main goal of this project is to preprocess data effectively to improve the accuracy of classification models, especially in the presence of imbalanced classes.

## Workflow
- Load and clean data  
- Remove outliers  
- Scale features (Standard or Min-Max Scaling)  
- Encode categorical variables  
- Handle class imbalance using SMOTE  
- Train and compare models (classification algorthims vs ANN)  
- Evaluate performance using accuracy, precision, recall, and F1-score

## Files Description
- **cleaned_data.csv**:  
  - Data cleaning performed  
  - Outliers removed  
  - Applied Standard Scaling  
  - Categorical variables transformed into numerical values

- **cleaned_r2.csv**:  
  - Scaling method changed from Standard Scaling to Min-Max Scaling

- **smote_file.py**:  
  - Addressed imbalanced data problem using SMOTE (Synthetic Minority Over-sampling Technique)

- **annfile.py**:  
  - Deep Learning model implemented using Artificial Neural Networks (ANN)
-**model1.py**:
   -random forest algorthem
-**model2.py**:
   -compare between classification algorthms

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score










