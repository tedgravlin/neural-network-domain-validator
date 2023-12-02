import pandas as pd
import joblib
from pyscript import document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from js import XMLHttpRequest
from js import window
import io
from os.path import exists
import os
import re

# Get the progress text HTML element
progress_text = document.querySelector("#progress-text")

# Gets the model and returns it
def load_model():

    progress_text.innerText = "Removing old model..."

    # Delete the model files if they already exist
    if (exists('model.pkl')):
        os.remove('model.pkl')
    if (exists('tfidf.pkl')):
        os.remove('tfidf.pkl')

    # Check if the current domain is an IP address (hosted locally) or a real domain (hosted on GH pages)
    if (re.search("^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$", window.location.hostname)):
        model_url = 'http://127.0.0.1:5500/models/model.pkl'
        tfidf_url = 'http://127.0.0.1:5500/models/tfidf.pkl'
    else:
        model_url = 'https://github.com/tedgravlin/neural-network-domain-validator/blob/a792b3c04d16be88eca95f3d6948d25f020722e2/models/model.pkl'
        tfidf_url = 'https://github.com/tedgravlin/neural-network-domain-validator/blob/a792b3c04d16be88eca95f3d6948d25f020722e2/models/tfidf.pkl'
    
    progress_text.innerText = "Fetching model.pkl..."

    # Get model.pkl
    model_req = XMLHttpRequest.new()
    model_req.open("GET", model_url, False)
    model_req.send(None)
    model_response = (model_req.response)

    progress_text.innerText = "Writing to model.pkl..."

    # Write the contents of the model response to model.pkl
    with open('model.pkl', 'w') as model_file:
        model_file.write(model_response)

    progress_text.innerText = "Fetching tfidf.pkl..."

    # Get tfidfk.pkl
    tfidf_req = XMLHttpRequest.new()
    tfidf_req.open("GET", tfidf_url, False)
    tfidf_req.send(None)
    tfidf_response = (tfidf_req.response)

    progress_text.innerText = "Writing to tfidf.pkl..."

    # Write the contents of the tfidf response to tfidf.pkl
    with open('tfidf.pkl', 'w') as tfidf_file:
        tfidf_file.write(tfidf_response)

    #model = joblib.load(model_file)
    #tfidf = joblib.load(tfidf_file)

    progress_text.innerText = "Model load complete."

    return exists('model.pkl'), exists('tfidf.pkl')

def test_model(model, tfidf):
    # Get the test dataset CSV file
    test_dataset = pd.read_csv("./dataset/testdataset.csv")

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
    print("ACTUAL:", y.to_list())
    print("CORRECT PREDICTIONS:", correct_count,"/",count)
    print("ACCURACY OF THIS TEST:", (correct_count/count) * 100)

def get_url(event):
    # Load the model from storage
    model = load_model()
    #test_model(model, tfidf)
    input_text = document.querySelector("#url-input")
    url = input_text.value
    output_text = document.querySelector("#result-text")
    output_text.innerText = model





    