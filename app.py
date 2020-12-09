"""
Created on Wed Dec  9 13:13:31 2020
@author: ISH KAPOOR
"""
import pandas as pd
import string
import nltk
import streamlit as st

data_file = 'C:/Users/ISH KAPOOR/Desktop/SMS-Spam-or-Ham-Filter-Predictor-main/SMSSpamCollection.txt'
data = pd.read_csv(data_file, sep = '\t', header = None, names = ['label', 'sms'])
#data.head()

nltk.download('stopwords')
nltk.download('punkt')

stopwords = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation

def pre_process(sms):

    lowercase = "".join([char.lower() for char in sms if char not in punctuation]) # Filters out punctuations and uppercase
    tokenize = nltk.tokenize.word_tokenize(lowercase)
    remove_stopwords = [word for word in tokenize if word not in stopwords]

    return remove_stopwords

data['processed'] = data['sms'].apply(lambda x: pre_process(x))
#data.head()

def categorize_words():

    spam_words = []
    ham_words = []

    for sms in data['processed'][data['label'] == 'spam']:
        for word in sms:
            spam_words.append(word)

    for sms in data['processed'][data['label'] == 'ham']:
        for word in sms:
            ham_words.append(word)

    return spam_words, ham_words

spam_words, ham_words = categorize_words()

def predict(user_input):

    spam_counter = 0
    ham_counter = 0

    for word in user_input:
        spam_counter += spam_words.count(word)
        ham_counter += ham_words.count(word)
    st.write("RESULTS:")
    if ham_counter > spam_counter:
        accuracy = round((ham_counter / (ham_counter + spam_counter)) * 100, 2)
        st.write("Message is not spam with {}% accuracy.".format(accuracy))
    elif ham_counter < spam_counter:
        accuracy = round((spam_counter / (ham_counter + spam_counter)) * 100, 2)
        st.write("Message is spam with {}% accuracy.".format(accuracy))
    else:
        st.write("It could be spam with 50% accuracy.")

menu = ["Home", "Predict Spam/Ham", "About"]
choice = st.sidebar.selectbox("Menu:", menu)

if choice == menu[0]:
    st.write('Welcome!')
elif choice == menu[1]:
    if st.checkbox("Upload Dataset"):
        dataset = st.file_uploader("Drop your spam/ham data set here", type=["csv"])
        if dataset is not None:
            data = pd.read_csv(dataset, sep = '\t', header = None, names = ['label', 'sms'])
    user_input = st.text_input("Enetr a Spam/Ham message:")
    processed_input = pre_process(user_input)
    prd = predict(processed_input)
else:
    st.write('Enjoy!')
