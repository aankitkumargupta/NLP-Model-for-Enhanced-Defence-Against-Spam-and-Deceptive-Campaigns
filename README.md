# NLP-Model-for-Enhanced-Defence-Against-Spam-and-Deceptive-Campaigns
## Overview
This repository contains a machine learning project that classifies text messages into spam or ham (non-spam). It uses various machine learning models like Naive Bayes, SVM, Logistic Regression, Random Forest, and KNN. The project includes a script for training models on a dataset, and a Flask application for deploying the models in a web interface.

## Dataset
The dataset used for training is the 'Spam/Ham Dataset'. It contains labeled messages categorized as spam or ham.

## Prerequisites
- Python 3.x
- Pandas
- NLTK
- scikit-learn
- joblib
- Flask

You can install the required packages using the following command: pip install pandas nltk scikit-learn joblib flask

## Project Structure
- `spam_ham_classifier.py`: Script to train and evaluate models.
- `app.py`: Flask application for deploying the models.
- `templates/`: Folder containing HTML templates for the Flask app.
- `model_accuracies.pkl`: File containing the accuracies of the trained models.
- `best_<model_name>.pkl`: Saved models after training.

### Running the Flask Application
1. Once the models are trained and saved, start the Flask application by running `app.py`.
2. Access the web interface through `http://127.0.0.1:5000` in your web browser.

## Usage
In the web application:
1. Enter the text message you want to classify.
2. Select the classification model.
3. Click on 'Classify' to get the prediction.

