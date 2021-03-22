# Disaster Response Pipeline (Udacity Data Science Nanodegree Project)

![](https://github.com/thiagogavioli/Disaster_Messages_Classifier/blob/main/screenshots/DRP_main.PNG)

## Table of contents
1. [Description](#Description)
2. [Installation](#Installation)
3. [Running the program](#Running_the_Program)
4. [File Descriptions](#File_Descriptions)
5. [Licensing, Authors, Acknowledgements](#Licensing-Authors-Acknowledgements)
6. [Screenshots](#Screenshots)

## Description

This project is part of Udacity's Data Science Nanodegree. It is in cooperation with Figure Eight, wich provides the necessary data to build a model for an API that classifies disaster messages. The original dataset contains real messages that were sent during disaster events. The goal of the project is to build a natural language processing (NLP) model to categorize messages on real time.

The project consists of:
1. An ETL pipeline to extract raw data, clean it and save in a SQLite DB
2. A machine learning pipeline to train a model able to classify text message into different categories
3. Run a web app that can show model results

## Installation

**The dependencies are listed below:**

- Python 3.5+
- Machine Learning Libraries: NumPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

## Running_the_program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File_Descriptions

**app/templates/:** html files for web app

**data/process_data.py:** ETL (extract transform and load) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

**models/train_classifier.py:** machine learning pipeline wich loads data, trains a model, and saves the trained model as a .pkl file

**run.py:** file used to start the Flask web app

**ETL Preparation Notebook:** step by step development of the ETL pipeline

**ML Pipeline Preparation Notebook:** step by step development of the machine learning pipeline

## Licensing_Authors_Acknowledgements

Must give credit to [Figure Eight](https://appen.com/) for the data used to train the model. The code can be used as any person demands.

## Screenshots

1. Some visualizations from the dataset
![](https://github.com/thiagogavioli/Disaster_Messages_Classifier/blob/main/screenshots/DRP_GraphOne.PNG)

![](https://github.com/thiagogavioli/Disaster_Messages_Classifier/blob/main/screenshots/DRP_GraphTwo.PNG)

![](https://github.com/thiagogavioli/Disaster_Messages_Classifier/blob/main/screenshots/DRP_Graphthree.PNG)

2. Example of a message to evaluate the model and its results
![](https://github.com/thiagogavioli/Disaster_Messages_Classifier/blob/main/screenshots/DRP_result.PNG)
