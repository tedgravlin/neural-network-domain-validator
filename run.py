import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('dataset.csv')
print("Dataset read")

#ohe = OneHotEncoder()
#transformed = ohe.fit_transform(dataset[['Label']])
#print(transformed.toarray())
#print(ohe.categories_)

y = dataset['Label']
z = dataset['URL']
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)

cv = CountVectorizer()
features = cv.fit_transform(z_train)
print("fit_transform completed")

model = svm.SVC()
model.fit(features,y_train)
print("model.fit completed")

features_test = cv.transform(z_test)
print("transform completed")

predictions = model.predict(features_test)
print('predicted', predictions)

print (classification_report(y_test, predictions))

print(confusion_matrix(y_test,predictions))

print("Accuracy: {}".format(model.score(features_test,y_test)))