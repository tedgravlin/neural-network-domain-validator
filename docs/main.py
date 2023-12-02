import pandas as pd
import joblib
from pyscript import document
from pyscript import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from js import XMLHttpRequest
#from js import window
import io
from os.path import exists
import os
#import re
import csv

# Get the progress text HTML element
progress_text = document.querySelector("#progress-text")
progress_text.innerText = "Pyscipt loaded."

# Gets the model and returns it
def load_model():

    progress_text.innerText = "Removing old model..."

    # Delete the model files if they already exist
    # if (exists('model.pkl')):
    #     os.remove('model.pkl')
    # if (exists('tfidf.pkl')):
    #     os.remove('tfidf.pkl')

    # model_url = 'http://127.0.0.1:5500/models/model.pkl'
    # tfidf_url = 'http://127.0.0.1:5500/models/tfidf.pkl'
    
    # progress_text.innerText = "Fetching model.pkl..."

    # # Get model.pkl
    # model_req = XMLHttpRequest.new()
    # model_req.open("GET", model_url, False)
    # model_req.send(None)
    # model_response = model_req.response.encode()

    # progress_text.innerText = "Writing to model.pkl..."

    # # Write the contents of the model response to model.pkl
    # with open('model.pkl', 'wb') as model_file:
    #     model_file.write(model_response)

    # progress_text.innerText = "Fetching tfidf.pkl..."

    # # Get tfidfk.pkl
    # tfidf_req = XMLHttpRequest.new()
    # tfidf_req.open("GET", tfidf_url, False)
    # tfidf_req.send(None)
    # tfidf_response = tfidf_req.response.encode()

    # progress_text.innerText = "Writing to tfidf.pkl..."

    # # Write the contents of the tfidf response to tfidf.pkl
    # with open('tfidf.pkl', 'wb') as tfidf_file:
    #     tfidf_file.write(tfidf_response)

    display("model file size", os.path.getsize("model.pkl"))
    display("tfidf file size", os.path.getsize("tfidf.pkl"))

    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')

    progress_text.innerText = "Model load complete."

    return model, tfidf

def test_model(model, tfidf):
    # Delete the test dataset if  it already exists
    if (exists('testdataset.csv')):
        os.remove('testdataset.csv')

    # Fetch testdataset.csv
    req = XMLHttpRequest.new()
    req.open("GET", 'http://127.0.0.1:5500/dataset/testdataset.csv',True)
    req.send(None)
    response = (req.response)

    # Write the contents of the testdataset response to testdataset.csv
    with open('testdataset.csv', 'w') as testdataset_file:
        testdataset_file.write(response)

    # Get the test dataset CSV file
    test_dataset = pd.read_csv("testdataset.csv")

    # Turn the test dataset into a pandas data frame
    dataframe = pd.DataFrame(test_dataset)
    x = dataframe[['Num Of Sections', 'TLD', 'TLD Length', 'Domain', 'Domain Length', 'URL']]
    y = dataframe['Label']

    # Separate text and numeric features
    text_features = ['TLD', 'Domain', 'URL']
    numeric_features = ['Num Of Sections', 'TLD Length', 'Domain Length']

    # Turn the text columns into a string and vectorize
    features_test_text = tfidf.transform(x[text_features].apply(lambda row: ' '.join(row.astype(str)), axis=1))

    # Scale the numeric features
    scaler = StandardScaler()
    scaler.fit_transform(x[numeric_features])
    features_test_numeric = scaler.transform(x[numeric_features])

    # Recombine the text_features and the numeric_features
    features_test = hstack([features_test_text, features_test_numeric])

    # Use the model to predict the label for each of 
    # the test domains and then print the result
    print("URLS:", x['URL'].to_list())
    prediction = model.predict(features_test)
    print("\nPREDIC:", prediction)  
    count = 0
    correct_count = 0
    for label in range(len(y)):
        if (prediction[label] == y.to_list()[label]):
            correct_count = correct_count + 1
        count = count + 1
    display("ACTUAL:", y.to_list())
    display("CORRECT PREDICTIONS:", correct_count,"/",count)
    display("ACCURACY OF THIS TEST:", (correct_count/count) * 100)

def get_url(event):
    # Load the model from storage
    model, tfidf = load_model()
    # Test the model with the test dataset
    test_model(model, tfidf)
    input_text = document.querySelector("#url-input")
    url = input_text.value
    output_text = document.querySelector("#result-text")
    #output_text.innerText = model





    