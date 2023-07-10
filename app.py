#import libraries
import streamlit as st
import pickle
import string
import sklearn as sk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

ps = PorterStemmer()

#transforming the text data
def transform_text(text):
    #convert text to lowercase
    text=text.lower()

    #tokenize text 
    text=nltk.word_tokenize(text)

    # remove special characters
    y=[]
    for word in text:
        if word.isalnum():
            y.append(word)

    #cloning the list
    text=y[:]
    y.clear()

    #remove stopwords
    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)
        
    #again cloning the list    
    text=y[:]
    y.clear()
    
    #stemming
    # ps = PorterStemmer()
    for word in text:
        y.append(ps.stem(word))
        

    return " ".join(y)

# Load the model from the file
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Title
st.title('SMS spam classifier')

input_text=st.text_area('Enter your message here')

if st.button('Predict'):
    # input_text = st.text_input('Enter your message here')
    # 1. Preprocess the text
    transform_sms = transform_text(input_text)
    # 2. Vectorize the text
    vector_input = tfidf.transform([transform_sms])
    # 3. Predict the text
    result = model.predict(vector_input)[0]
    # 4. Display the results 
    if result == '1':
        st.header('Spam message')
    else:
        st.header('Not spam message')
