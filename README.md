# Disaster Response Pipeline Project

## Table of contents
* [Introduction](#Introduction)
* [FileDescriptions](#FileDescriptions)
* [About](#About)

## Introduction
This project analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages. This projects contains three parts:
* Data Processing: Read the dataset, clean the data, and then store it in a SQLite database.
* Machine Learning Pipeline: Built machine learning pipeline to train and classify the data. 
* Web-app Deploy: Deploy the results.

## File Descriptions
- Data
  - DisasterResponse.db: The database created from data processing.
  - disaster_categories.csv & disaster_messages.csv: Original dataset.
  - process_data.py: ETL pipeline that processed the data.
  
- Models
  - train_classifier.py: ML pipeline. Split the data into a training set and a test set. Then, create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and RandomSearchCV to output a final model that predicts a message classifications for the 36 categories (multi-output classification)
  - classifier.pkl: The ML model.
- App
  - run.py: Deploy the model.
  
## About

This project is a part of Udacity Data Scientist NanoDegree programme. The data was provided by Figure Eight. 
