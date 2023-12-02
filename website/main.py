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
import io
from os.path import exists
import os

# Gets the model and returns it
def load_model():
    model_url = 'http://127.0.0.1:5500/models/model.pkl'
    req = XMLHttpRequest.new()
    req.open("GET", model_url, False)
    req.send(None)
    model_response = (req.response)

    if (exists('model.pkl')):
        # Delete model.pkl
        os.remove('model.pkl')

    if (exists('tfidf.pkl')):
        # Delete model.pkl
        os.remove('tfidf.pkl')

    # Write the contents of the model response to model.pkl
    with open('model.pkl', 'w') as f:
        f.write(model_response)

    model_file_exists = exists('model.pkl')

    model_url = 'http://127.0.0.1:5500/models/tfidf.pkl'
    tfreq = XMLHttpRequest.new()
    tfreq.open("GET", model_url, False)
    tfreq.send(None)
    tfidf_response = (tfreq.response)
    # Write the contents of the tfidf response to tfidf.pkl
    with open('tfidf.pkl', 'w') as f:
        f.write(tfidf_response)

    tfidf_file_exists = exists('tfidf.pkl')

    #model = joblib.load(model_file)
    #tfidf = joblib.load(tfidf_file)
    return model_file_exists, tfidf_file_exists

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
    #output_text.innerText = url
    output_text.innerText = model






    