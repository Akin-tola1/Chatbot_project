import nltk
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

data = pd.read_csv('Samsung Dialog.txt', sep= ':',header= None)
data.head()

cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0] == 'Sales Agent']

df = pd.DataFrame()

df['Questions'] = cust[1].reset_index(drop = True)
df['Answer'] = sales[1].reset_index(drop = True)

df.head()


def preprocess_text(text):
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)

    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
      
        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)

    return ' '.join(preprocessed_sentences)


df['tokenized Questions'] = df['Questions'].apply(preprocess_text)

corpus = df['tokenized Questions'].to_list()

tfidf_vectorizer = TfidfVectorizer()

vectorised_corpus = tfidf_vectorizer.fit_transform(corpus)





st.markdown("<h1 style = 'text-align: center; color: #176B87'>SAMSUNG CUSTOMER CARE</h1>", unsafe_allow_html = True)
st.markdown("<h6 style = 'text-align: center; top-margin: 0rem; color: #64CCC5'>BUILT BY DEJI</h1>", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)
col1, col2 = st.columns(2)
col1.image('pngwing.com (5).png', caption = 'Phone Related Chats')


def bot_response(user_input):
    user_input_processed = preprocess_text(user_input)
    v_input = tfidf_vectorizer.transform([user_input_processed])
    most_similar = cosine_similarity(v_input, vectorised_corpus)
    most_similar_index = most_similar.argmax()
    
    return df['Answer'].iloc[most_similar_index]

chatbot_greeting = [
    "Hello there, welcome to DEJI'S Bot. pls enjoy your usage",
    "Hi user, This bot is created by DEJI, enjoy your usage",
    "Hi hi, How you dey my nigga",
    "Alaye mi, Abeg enjoy your usage",
    "Hey Hey, pls enjoy your usage"    
]

user_greeting = ["hi", "hello there", "hey", "hi there"]
exit_word = ['bye', 'thanks bye', 'exit', 'goodbye']


user_q = col2.text_input('Pls ask your phone related question: ')
if user_q in user_greeting:
    col2.write(random.choice(chatbot_greeting))
elif user_q in exit_word:
    col2.write('Thank you for your usage. Bye')
elif user_q == '':
    st.write('')
else:
    responses = bot_response(user_q)
    col2.write(f'ChatBot:  {responses}')
