#importnt needed libararies
import streamlit as st
import numpy as np
import pandas as pd
import re
#import matplotlib 
import plotly.express as px
import nltk
import matplotlib.pyplot as plt 
import seaborn as sns 
#import spacy
import string
from PIL import Image
pd.options.mode.chained_assignment = None  #handling arnings by pandas dataframe

#read data
df= pd.read_csv("C:/Users/ahmed pc/Data Science/Assignment Text Analysis/SMS_data.csv", encoding = 'unicode_escape')
df['Message_body'] = df['Message_body'].astype(str) #type string 
#Lower casing
df['Message_updated'] = df['Message_body'].str.lower()
#drop message column
df.drop(["Message_body"], axis=1, inplace=True)

#Remove punctuation
PUNCT = string.punctuation
#function to remove punctuation
def remove_punct (message_updated):
    return message_updated.translate(str.maketrans('' , '' , PUNCT))
#call function
df['Message_updated'] = df['Message_updated'].apply(lambda Message_body: remove_punct(Message_body))

#removal of stopwords
#download stopwords from nltk
nltk.download('stopwords')
#import library
from nltk.corpus import stopwords
#store all stopwords in variable
STOPWORDS = set(stopwords.words('english'))
#fuction to remove stopwords
def remove_stopwords (Message_updated):
    return ' '.join([word for word in str(Message_updated).split() if word not in STOPWORDS])
#call function
df['Message_updated'] = df['Message_updated'].apply(lambda Message_updated:remove_stopwords (Message_updated))

#remove numbers

#function to remove numbers
def remove_numbers(Message_updated):
    numb_pattern = re.compile(r'[0-9]+')
    return numb_pattern.sub('', Message_updated)
#call function
df['Message_updated'] = df['Message_updated'].apply(lambda Message_updated: remove_numbers(Message_updated))

#Remove URLs
#function to remove urls
def remove_urls(Message_updated):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', Message_updated)
#call function
df['Message_updated'] = df['Message_updated'].apply(lambda Message_updated: remove_urls(Message_updated))

#Lemmitization
#import libraries
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(Message_updated):
    pos_tagged_text = nltk.pos_tag(Message_updated.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

df['Message_updated'] = df['Message_updated'].apply(lambda Message_updated: lemmatize_words(Message_updated))

#convert date column to date time type
import datetime
df['Date_Received'] = pd.to_datetime(df['Date_Received'])

#spam messages data
spam_Msg = df[(df['Label'] == 'Spam')]
df1 = spam_Msg
df1.head()
#Non _ spam messages data
Non_spam_Msg = df[(df['Label'] == 'Non-Spam')]
df2 = Non_spam_Msg
df2.head()

#Extract Most common keywords
from collections import Counter
cnt = Counter()
for Message_updated in df1['Message_updated'].values:
    for word in Message_updated.split():
        cnt[word] += 1
sc = cnt.most_common(15)        
spam_count = pd.DataFrame(sc, columns = ['KeyWords' , 'Counts'])

#Extract the most common keywords of Non_spam messages
cnt = Counter()
for Message_updated in df2['Message_updated'].values:
    for word in Message_updated.split():
        cnt[word] += 1        
nsc =  cnt.most_common(15)
nspam_count = pd.DataFrame(nsc ,columns=['KeyWords','Counts'])

#Number of Messages Received over Date
msg =df.groupby('Date_Received')['Message_updated'].count() 
Msg_counts = pd.DataFrame(msg)

def main():
    st.title('SMS Web App')
    image = Image.open('E:/image.jpg')
    st.image(image, use_column_width = True)
    st.header('Created by: Waliya Shah ')
    sm = st.selectbox('pleae Select Message type:' ,['Spam' , 'Non_Spam'])
   # show = st.button('Show')
    #if show:      
    if sm == 'Spam':
        st.markdown('Spam Messages Data')
        st.dataframe(df1.head())
        button = st.button('Show Visualization')
        if button:
            b1,b2 =st.columns(2)
            with b1:
                fig=px.bar(spam_count,x ='Counts',y = 'KeyWords' 
                           ,title ='Most Common Keywords within Spam SMS',
                       height = 500 ,width = 550 , orientation='h' )
                st.write(fig)
            with b2:
                fig = px.line(Msg_counts, #Data Frame
                             title = "Number of Messages Received over Dates")
                st.write(fig)
               
           #st.line_chart(Msg_counts)
    else:
        st.markdown('Non_Spam Messages Data')
        st.dataframe(df2.head())
        button = st.button('Show Visualization')
        if button:
            c1 ,c2 = st.columns(2)
            with c1:
                fig=px.bar(nspam_count,x ='Counts',y = 'KeyWords' ,
                           title ='Most Common Keywords within Non_Spam SMS',
                           height = 500 ,width = 550 , orientation='h' )
                st.write(fig)
                   #fig.show()
                   #st.bar_chart(nspam_count['Counts'])
            with c2:
                fig = px.line(Msg_counts, #Data Frame
                      title = "Number of Messages Received over Dates")
                st.write(fig)
        
        
if __name__ == '__main__':
    main()