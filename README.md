# Disaster Response Pipeline Project

### Summary
In this project, I have analyzed disaster data from Appen to build a model for an API that classifies disaster messages.
There is a data set containing real messages that were sent during disaster events. I have created a machine learning pipeline to categorize these events so that I can send the messages to an appropriate disaster relief agency.
My project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

### Instructions to run the code:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterMessages.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterMessages.db models/my_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
