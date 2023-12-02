import pandas as pd
import joblib
from pyscript import document
from pyscript import display
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import os
from pyweb import pydom

# Get the input container element
input_container = pydom['#input-container']
input_container.style["display"] = "block"

# Get the progress text HTML element
progress_text = document.querySelector("#progress-text")
progress_text.innerText = ""

# Gets the model and returns it
def load_files():
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    test_dataset = pd.read_csv("testdataset.csv")

    progress_text.innerText = "Model load complete."

    return model, tfidf, test_dataset

def test_model(model, tfidf, test_dataset):
    progress_text.innerText = "Testing URL against model..."

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
    display("URLS:", x['URL'].to_list())
    prediction = model.predict(features_test)
    display("\nPREDIC:", prediction)  
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
    input_text = document.querySelector("#url-input")
    url = input_text.value
    output_text = document.querySelector("#result-text")
    output_text.innerText = url
    run(url)

def run(url):
    # Load the model from storage
    model, tfidf, test_dataset = load_files()
    # Test the model with the test dataset
    test_model(model, tfidf, test_dataset)





    