import pandas as pd
import joblib
from pyscript import document
from pyscript import display
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from pyweb import pydom
from urllib.parse import urlparse
import csv

# Get the input container element
input_container = pydom['#input-container']
input_container.style["display"] = "block"

# Get the progress text HTML element
progress_text = document.querySelector("#progress-text")
progress_text.innerText = ""

# Get the result text HTML element
result_text = document.querySelector('#result-text')

# Gets the model and returns it
def load_files():
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')

    return model, tfidf

def test_model(model, tfidf, formatted_url):
    with open('testdataset.csv', mode ='w', newline='', encoding="utf-8") as test_file:
        parameters = []
        # Open writer to CSV file
        csvWrite = csv.writer(test_file, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
        # Add the labels to parameters
        parameters.append('NumOfSections,TLD,TLDLength,Domain,DomainLength,URL')
        # Write the labels to the CSV file
        csvWrite.writerow(parameters)
        parameters = []
        # Add the URL to parameters
        parameters.append(formatted_url)
        csvWrite.writerow(parameters)

    test_dataset = pd.read_csv("testdataset.csv")

    # Turn the test dataset into a pandas data frame
    dataframe = pd.DataFrame(test_dataset)
    x = dataframe[['NumOfSections', 'TLD', 'TLDLength', 'Domain', 'DomainLength', 'URL']]

    # Separate text and numeric features
    text_features = ['TLD', 'Domain', 'URL']
    numeric_features = ['NumOfSections', 'TLDLength', 'DomainLength']

    # Turn the text columns into a string and vectorize
    features_test_text = tfidf.transform(x[text_features].apply(lambda row: ' '.join(row.astype(str)), axis=1))

    # Scale the numeric features
    scaler = StandardScaler()
    scaler.fit_transform(x[numeric_features])
    features_test_numeric = scaler.transform(x[numeric_features])

    # Recombine the text_features and the numeric_features
    features_test = hstack([features_test_text, features_test_numeric])

    # Use the model to predict the label for the url
    prediction = model.predict(features_test)
    result_text.innerText = "Prediction: " + prediction[0]
    

def get_url(event):
    input_text = document.querySelector("#url-input")
    url = input_text.value
    run(get_formatted_url(url))

def get_hostname(url):
    if (urlparse(url).scheme == ""):
        url = "https://" + url

    result = urlparse(url)
    return result.hostname

def get_formatted_url(url):
    link = get_hostname(url)
    comma = ","
    # Split the link into sections
    linkSplit = link.split(".")
    # Get the number of sections
    numOfSections = len(linkSplit)
    # Get the TLD and its length
    tld = linkSplit[len(linkSplit) - 1]
    tldLength = len(tld)
    # Get the domain and its length
    domain = linkSplit[len(linkSplit) - 2]
    domainLength = len(domain)
    return (str(numOfSections) + comma + tld + comma + str(tldLength) + comma + domain + comma + str(domainLength) + comma + link)

def run(url):
    model, tfidf = load_files()
    test_model(model, tfidf, url)





    