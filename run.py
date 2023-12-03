import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)

# FUNCTION: Train the model and store it locally
def create_model():
    # Get the dataset CSV file
    dataset = pd.read_csv("./dataset/dataset.csv")

    # Turn the dataset into a pandas data frame
    dataframe = pd.DataFrame(dataset)
    x = dataframe[['Num Of Sections', 'TLD', 'TLD Length', 'Domain', 'Domain Length', 'URL']]
    y = dataframe['Label']

    # Separate text and numeric features
    text_features = ['TLD', 'Domain', 'URL']
    numeric_features = ['Num Of Sections', 'TLD Length', 'Domain Length']

    # Split training and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

    # Initialize the TF-IDF vectorizer
    tfidf = TfidfVectorizer(min_df=1)

    # Turn the text columns into a string and vectorize
    features_train_text = tfidf.fit_transform(x_train[text_features].apply(lambda row: ' '.join(row.astype(str)), axis=1))
    features_test_text = tfidf.transform(x_test[text_features].apply(lambda row: ' '.join(row.astype(str)), axis=1))

    # Scale the numeric features
    scaler = StandardScaler()
    features_train_numeric = scaler.fit_transform(x_train[numeric_features])
    features_test_numeric = scaler.transform(x_test[numeric_features])

    # Recombine the text_features and the numeric_features
    features_train = hstack([features_train_text, features_train_numeric])
    features_test = hstack([features_test_text, features_test_numeric])

    # MLP model
    model = MLPClassifier(
        activation='relu',
        batch_size=128,
        hidden_layer_sizes=(256),
        learning_rate='adaptive',
        verbose=True,
    )

    # The number of epochs to run
    N_EPOCHS = 10
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('Epoch: ', epoch)
        model.partial_fit(features_train, y_train, classes=N_CLASSES)
            
        # SCORE TRAIN
        scores_train.append(model.score(features_train, y_train))

        # SCORE TEST
        scores_test.append(model.score(features_test, y_test))

        epoch += 1

    # Plot accuracy over iterations
    plt.plot(scores_train, color='green', alpha=0.8, label='Train')
    plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.show()

    # Predicting the test set results
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

# FUNCTION: Grab the model from storage and test it with some domains
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

# FUNCTION: Delete the model from storage
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


# TODO: Add .edu domains to the dataset