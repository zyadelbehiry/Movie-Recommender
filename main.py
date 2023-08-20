# important installing

# import liberaries
import os
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn import model_selection, preprocessing, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# function that takes the folder names as a parameter
def read_data(folders):
    # initialize an empty list to store the data
    data = []
    # loop through the folders
    for folder in folders:
        # join the folder name with the file path
        folder_path = os.path.join('C:\\Users\\win10\\Desktop\\review_polarity\\txt_sentoken', folder)
        # loop through the files in the folder
        for file in os.listdir(folder_path):
            # join the file name with the folder path
            file_path = os.path.join(folder_path, file)
            # open the file and read its contents
            with open(file_path, 'r') as f:
                text = f.read()
            # assign the label as the folder name
            label = folder
            # append a tuple of text and label to the data list
            data.append((text, label))
    # convert the data list to a pandas dataframe
    df = pd.DataFrame(data, columns=['text', 'label'])
    # return the dataframe
    return df


# call the function with the desired arguments
folders = ['neg', 'pos']
df = read_data(folders)


# function that takes the dataframe and the test size as parameters
def preprocess_data(df, test_size):
    # perform text preprocessing

    # convert to lowercase
    df['text'] = df['text'].apply(lambda x: x.lower())

    # remove punctuations
    # ^->not , \W ->Word , \s -> white space
    df['text'] = df['text'].str.replace('[^\w\s]', '', regex=True)

    # remove stopwords
    stop = stopwords.words('english')
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # stem words
    st = SnowballStemmer()
    df['text'] = df['text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

    # shuffle and split data into training and validation sets
    df = df.sample(frac=1, random_state=42)
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['text'], df['label'], test_size=test_size,
                                                                          random_state=42)

    # encode labels
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.transform(valid_y)

    # return the split and encoded data
    return train_x, valid_x, train_y, valid_y


# call the function with the desired arguments
test_size = 0.2
train_x, valid_x, train_y, valid_y = preprocess_data(df, test_size)


# define a function that takes the dataframe and the training and validation data as parameters
def extract_features(df, train_x, valid_x):
    # create a TF-IDF vectorizer object
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    # fit the vectorizer on the dataframe text column
    tfidf_vect.fit(df['text'])
    # transform the training and validation data using the vectorizer
    xtrain_tfidf = tfidf_vect.transform(train_x)
    xvalid_tfidf = tfidf_vect.transform(valid_x)
    # return the transformed data
    return xtrain_tfidf, xvalid_tfidf


# call the function with the desired arguments
xtrain_tfidf, xvalid_tfidf = extract_features(df, train_x, valid_x)


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions, valid_y)


# Naive Bayes trainig
accuracy1 = train_model(naive_bayes.MultinomialNB(alpha=0.6), xtrain_tfidf, train_y, xvalid_tfidf)
print("Accuracy of naive_bayes classifier is: ", accuracy1)

accuracy2 = train_model(svm.SVC(kernel='linear'), xtrain_tfidf, train_y, xvalid_tfidf)
print("Accuracy of svm classifier is: ", accuracy2)

accuracy3 = train_model(RandomForestClassifier(n_estimators=100), xtrain_tfidf, train_y, xvalid_tfidf)
print("Accuracy of RandomForest classifier is: ", accuracy3)

from sklearn.ensemble import GradientBoostingClassifier

accuracy4 = train_model(GradientBoostingClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print("Accuracy of GradientBoosting classifier is: ", accuracy4)

from sklearn.tree import DecisionTreeClassifier

accuracy5 = train_model(DecisionTreeClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print("Accuracy of DecisionTree classifier is: ", accuracy5)


def print_best_model(models, accuracies):
    best_index = accuracies.index(max(accuracies))
    best_model = models[best_index]
    best_accuracy = accuracies[best_index]
    print(f"The best model is {best_model} with an accuracy of {best_accuracy}")


models = ['Naive Bayes', 'SVM', 'Random Forest', 'Gradient Boosting', 'Decision Tree']
accuracies = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5]

print_best_model(models, accuracies)

# from sklearn.neighbors import KNeighborsClassifier
# accuracy = train_model(KNeighborsClassifier(n_neighbors=9), xtrain_tfidf, train_y, xvalid_tfidf)
# print ("Accuracy of KNeighbors classifier is: ", accuracy)
import matplotlib.pyplot as plt

# define the classifiers and their names
classifiers = [
    ('Naive Bayes', naive_bayes.MultinomialNB(alpha=0.6)),
    ('SVM', svm.SVC(kernel='linear')),
    ('Random Forest', RandomForestClassifier(n_estimators=100)),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('Decision Tree', DecisionTreeClassifier())
]

# initialize lists to store the accuracies
train_accs = []
test_accs = []

# loop through the classifiers
for name, clf in classifiers:
    # train the classifier on the training data
    clf.fit(xtrain_tfidf, train_y)
    # calculate the training accuracy
    train_acc = clf.score(xtrain_tfidf, train_y)
    train_accs.append(train_acc)
    # calculate the testing accuracy
    test_acc = clf.score(xvalid_tfidf, valid_y)
    test_accs.append(test_acc)

# create a bar chart to visualize the accuracies
x_pos = [i for i, _ in enumerate(classifiers)]
plt.bar(x_pos, train_accs, color='blue', alpha=0.5, label='Training Accuracy')
plt.bar(x_pos, test_accs, color='red', alpha=0.5, label='Testing Accuracy')
plt.xticks(x_pos, [name for name, _ in classifiers])
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


from sklearn.metrics import confusion_matrix

# get the predictions for the validation set using the best model
best_model = svm.SVC(kernel='linear')
best_model.fit(xtrain_tfidf, train_y)
y_pred = best_model.predict(xvalid_tfidf)

# calculate the confusion matrix
cm = confusion_matrix(valid_y, y_pred)

# plot the confusion matrix
plot_confusion_matrix(cm, classes=['neg', 'pos'], normalize=True, title='Confusion Matrix')

