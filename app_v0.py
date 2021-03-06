################# LIBRARIES #############
import tensorflow as tf 
import numpy as np
import pandas as pd
import random
import csv
# Streamlit
import streamlit as st
# import model
import joblib
# sql connection
import pyodbc
# azure cognitive service connection
import azure.cognitiveservices.speech as speechsdk
# google translate API
from google_trans_new import google_translator
# NLP
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Models

model_text_class = tf.keras.models.load_model('models/texte_classification.h5')
model_carqiaque = joblib.load("models/model_cardiaque.pkl")

####### Model Text Classifier Requirements #############
num_sentences = 0
corpus = []

with open("datasets/text_classify_2_columns_data.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        list_item=[]
        list_item.append(row[0])
        this_label=row[1]
        if this_label=='Emotional pain':
            list_item.append(0)
        elif this_label=='Hair falling out':
            list_item.append(1)
        elif this_label=='Heart hurts':
            list_item.append(2)
        elif this_label=='Infected wound':
            list_item.append(3)
        elif this_label=='Foot ache':
            list_item.append(4)
        elif this_label=='Shoulder pain':
            list_item.append(5)
        elif this_label=='Injury from sports':
            list_item.append(6)
        elif this_label=='Skin issue':
            list_item.append(7)
        elif this_label=='Stomach ache':
            list_item.append(8)
        elif this_label=='Knee pain':
            list_item.append(9)
        elif this_label=='Joint pain':
            list_item.append(10)
        elif this_label=='Hard to breath':
            list_item.append(11)
        elif this_label=='Head ache':
            list_item.append(12)
        elif this_label=='Body feels weak':
            list_item.append(13)
        elif this_label=='Feeling dizzy':
            list_item.append(14)
        elif this_label=='Back pain':
            list_item.append(15)
        elif this_label=='Open wound':
            list_item.append(16)
        elif this_label=='Internal pain':
            list_item.append(17)
        elif this_label=='Blurry vision':
            list_item.append(18)
        elif this_label=='Acne':
            list_item.append(19)
        elif this_label=='Muscle pain':
            list_item.append(20)
        elif this_label=='Neck pain':
            list_item.append(21)
        elif this_label=='Cough':
            list_item.append(22)
        elif this_label=='Ear ache':
            list_item.append(23)
        else:
            list_item.append(24)
        num_sentences = num_sentences + 1
        corpus.append(list_item)

#oov_tok = "<OOV>" 
#training_size=6661

sentences=[]
labels=[]
random.shuffle(corpus)
for x in range(6661):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

def text_classify(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=30, padding='post', truncating='post')
    model_text_class.predict(padded)
    score = np.amax(model_text_class.predict(padded))
    classs = np.where(model_text_class.predict(padded) == np.amax(model_text_class.predict(padded)))[1][0]
    return score, classs

dict_class = {0:'Emotional pain', 1:'Hair falling out',2:'Heart hurts',
      3:'Infected wound',4:'Foot ache',5:'Shoulder pain',6:'Injury from sports',
      7:'Skin issue',8:'Stomach ache',9:'Knee pain',10:'Joint pain',11:'Hard to breath',
      12:'Head ache',13:'Body feels weak',14:'Feeling dizzy',15:'Back pain',
       16:'Open wound',17:'Internal pain',18:'Blurry vision',19:'Acne',
      20:'Muscle pain',21:'Neck pain',22:'Cough',23:'Ear ache',24:'Feeling cold'}

####### SQL Connection Variables #############
server = "serversqlhak.database.windows.net"
database = "sql_db_hak"
username = "hak"
password = "VzrpT2jWpy7Z2NZ"
driver = "{ODBC Driver 17 for SQL Server}"

####### SQL Connection Configuration #############
cnxn = pyodbc.connect(
            "DRIVER="
            + driver
            + ";PORT=1433;SERVER="
            + server
            + ";PORT=1443;DATABASE="
            + database
            + ";UID="
            + username
            + ";PWD="
            + password
        )
cursor = cnxn.cursor()

####### Functions #############
# Speach to Text
def from_mic():
    speech_config = speechsdk.SpeechConfig(
        subscription="c3cedf0693534b09a7914bf6b57a97f5", region="westeurope"
    )
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, language="fr-FR"
    )
    result = speech_recognizer.recognize_once_async().get()
    return result

# Translatation
def translatoor(origin):
    translator = google_translator()
    text_en = translator.translate(origin, lang_tgt="en")
    return text_en

####### SQL Connection Configuration #############
text_fr = ''
text_en = ''

def main():

    global text_fr
    global text_en
    
    st.markdown("<style>h1{color: red;}</style>", unsafe_allow_html=True)
    st.markdown('# <div align="center">Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '## <div align="center">**Analysis**</div>',
        unsafe_allow_html=True,
    )
    #st.markdown(" ")
    #st.markdown(" ")

    #st.sidebar.title("INFO PATIENT")

    #st.subheader("PATIENT ou MEDECIN")
    pat_med = st.selectbox("Patient ou Medecin", ("Patient", "Medecin"))

    if pat_med == "Patient":
        st.markdown("## Page du Patient")

        tension = st.number_input("Tension", 70, 130, step=1, key="tension")
        if tension > 75:
            st.success("ALERT!! YOU ARE DYING")


        if st.button("PARLE-TOI"):
            result = from_mic()
            text_fr = result.text

            #text_en = translatoor(text_fr)
            #st.write("Transcription from Speech")
            st.success(text_fr)
            #st.write("Translation from French to English")
            #st.success(text_en)

            #model_text_class = tf.keras.models.load_model('models/texte_classification.h5')

            #model_alex = joblib.load("models/speach_label.pkl")
            #pred = model_alex.predict()
            #proba = model_alex.predict_proba()
            cursor.execute(
            "insert into dbo.sentences2(Sentences, Label) values (?, ?)",
            f"{text_fr}",
            "Heart Attack")
            cnxn.commit()

    if pat_med == "Medecin":
        
        st.markdown("## Page du Medecin")
        st.sidebar.title("INFO PATIENT")
        ejection_fraction = st.sidebar.number_input(
            "Ejection Fraction", 10, 100, step=1, key="ejection_fraction"
        )

        serum_creatinine = st.sidebar.number_input(
            "Serum Creatinine", 0.1, 10.0, step=0.1, key="serum_creatinine"
        )

        time = st.sidebar.number_input("Time", 1, 300, step=1, key="time")

        #sex = st.sidebar.selectbox("Sex", ("Femme", "Homme"))

        #age = st.sidebar.number_input("Age", 1, 130, step=1, key="age")
        
        if st.button("Calcule"):
            cursor.execute(
            "SELECT TOP 1 Sentences FROM dbo.sentences2 ORDER BY Id DESC" )
            row = cursor.fetchone()
            cnxn.commit()
            st.write('Phrase')
            st.success(row[0])
            
            text_en = translatoor(row[0])
            scorr, clas = text_classify(text_en)
            st.write('Detected Class')
            st.success(dict_class[clas])
            st.write('Probability')
            st.success("{} %".format(round(scorr*100, 2)))
            #st.success(scorr)

            X_new = np.array([[ejection_fraction, serum_creatinine, time]])
            #model_carqiaque = joblib.load("models/model_cardiaque.pkl")
            pred = model_carqiaque.predict(X_new)[0]
            proba = model_carqiaque.predict_proba(X_new)[0][1]
            st.write("MORT")
            st.success(pred)
            st.write("Probability")
            st.success(proba)

if __name__ == "__main__":
    main()