import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
import os

def create_model():
    # Get the dataset CSV file
    dataset = pd.read_csv("dataset.csv")

    # Turn the dataset into a pandas data frame
    dataframe = pd.DataFrame(dataset)

    # Dataset columns
    x = dataframe['URL']
    y = dataframe['Label']

    #Split train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Initialize the TF-IDF vectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
    stop_words='english')

    # Generate features for x_train and x_test
    features_train = tfidf.fit_transform(x_train)
    features_test = tfidf.transform(x_test)

    # MLP model
    model = MLPClassifier(
        activation='relu',
        batch_size=50,
        hidden_layer_sizes=(256),
        random_state=42,
        learning_rate='constant',
        learning_rate_init=0.001,
        verbose=True,
    )

    # Fit the model
    model.fit(features_train, y_train)

    # Predicting the Test set results
    predictions = model.predict(features_test)

    # Print the classification report
    print("Classification Report")
    print(classification_report(y_test, predictions))

    # Print the confusion matrix
    print("Confusion Matrix")
    print(confusion_matrix(y_test,predictions))

    # Print the model accuracy
    print("Accuracy: ", model.score(features_test,y_test) * 100)

    # Store the model in storage
    joblib.dump(model,"./models/model.pkl")
    # Store the vectorizer in storage
    joblib.dump(tfidf, "./models/tfidf.pkl")

def test_model(model, tfidf):
    # Test domains
    test_domains = ['google.com','googl.com']

    # Use the model to predict the label for each of 
    # the test domains and then print the result
    for domain in range(len(test_domains)):
        print(test_domains[domain])
        test_result = tfidf.transform([test_domains[domain]])
        prediction = model.predict(test_result)
        print("PREDICTION:", prediction)  

def erase_model():
    os.remove("./models/model.pkl")
    os.remove("./models/tfidf.pkl")  

# If there's a model in storage
if (os.path.exists('./models/model.pkl')):
    # Ask the user if they want to use the stored model
    model_choice = input("There's already a model stored. Would you like to use it? (Y/N): ").lower()

    # If the user chooses NOT to use the stored model
    if (model_choice == 'n'):
        # Tell user that old model will be erased and new one will be created
        confirmation = input("Continuing will erase the existing model. Do you want to continue? (Y/N): ").lower()
        if (confirmation == 'y'):
            print("Erased stored model. Creating new one...")
            erase_model()
            create_model()

    # If the user chooses to use the stored model
    if (model_choice == 'y' or confirmation =='n'):
        # Load the model from storage
        model = joblib.load("./models/model.pkl")
        tfidf = joblib.load("./models/tfidf.pkl")
        print ("Loaded model from storage.")
        # Test the model
        test_model(model, tfidf)
# Else, create a new model.
else:
    # Print message to reflect that no model is stored
    print("No model stored. Creating new one...")
    create_model()
