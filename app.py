"""
Created on Wed Dec  9 13:13:31 2020
@author: ISH KAPOOR
"""
import pandas as pd
import string
import nltk
import streamlit as st

data_file = 'SMSSpamCollection.txt'
data = pd.read_csv(data_file, sep = '\t', header = None, names = ['label', 'sms'])
#data.head()
info1 = '''
Have you ever received a text message from a company or person you recognized, but it didn’t seem quite “right”? 

Spam text messages (also known as phishing texts or “smishing” – SMS phishing) are tools criminals use to trick you into giving them your personal or financial information. 

Criminals use phishing text messages to attain usernames and passwords, social security numbers, credit card numbers and PINs to commit fraud or identity theft. Other attacks focus on duping people into downloading viruses or malware by clicking seemingly innocent links.
'''
info2 = '''
SMS spam (sometimes called cell phone spam) is any junk message delivered to a mobile phone as text messaging through the Short Message Service (SMS). the systems of the country's major service provider, were overcome by the volume of SMS spam, causing users' screens to freeze and spreading programs that caused the phones to dial emergency numbers.
Text messaging has greatly increased in popularity in the past five years and the government is trying to keep up with rapidly changing technology. Although SMS spam is less prevalent than email spam, it still accounts for roughly 1% of texts sent in the United States and 30% of text messages sent in parts of Asia. In the United States, SMS spam messages have been illegal under the Telephone Consumer Protection Act since 2004.
Spam legislation is non-existent in India. The much-touted Information Technology Act of 2000 does not discuss the issue of spamming at all. It only refers to punishment meted out to a person, who after having secured access to any electronic material without the consent of the person concerned, discloses such electronic material to any other person. It does not have any bearing on violation of individual's privacy in Cyberspace. The illegality of spamming is not considered.
Citizens who receive unsolicited SMS messages can now bring the solicitors to small claims court. In 2009, China’s three main mobile phone operators (China Telecom, China Mobile Ltd and China Unicom) signed an agreement to combat mobile spam by setting limits on the number of text messages sent each hour.
'''

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
    st.write("""
#    Welcome!
    """)
    st.header("Spam text messages/SMS")
    st.write(info2)
    st.write(info1)
elif choice == menu[1]:
    if st.checkbox("Upload Dataset"):
        dataset = st.file_uploader("Drop your spam/ham data set here", type=["csv"])
        if dataset is not None:
            data = pd.read_csv(dataset, sep = '\t', header = None, names = ['label', 'sms'])
    user_input = st.text_input("Enetr a Spam/Ham message:")
    processed_input = pre_process(user_input)
    prd = predict(processed_input)
else:
    st.markdown("# Enjoy!")
    st.markdown("[![this is an image link](https://discourse-cloud-file-uploads.s3.dualstack.us-west-2.amazonaws.com/business7/uploads/streamlit/original/1X/00ddf47b5bf61fb18e1c36fc6680b1ee0c7778fb.png)]")
    st.write("""
    # E njoy!
    """)
    st.write("""
    ## View source code at:
    """)
    st.write("https://github.com/ishkapoor2000/SMS-Spam-or-Ham-Filter-Predictor")
