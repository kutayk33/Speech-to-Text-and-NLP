################# LIBRARIES #############
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import azure.cognitiveservices.speech as speechsdk
from google_trans_new import google_translator
import random, csv, joblib, pyodbc
import tensorflow as tf 
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

####### Model Text Classifier Requirements #############
@st.cache(persist=True)
def get_tokenizer():

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
    return tokenizer

#Model Text Classifier
def text_classify(text):
    tokenizer = get_tokenizer()
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=30, padding='post', truncating='post')
    #model_text_class, _ = get_models()
    model_text_class = tf.keras.models.load_model('models/texte_classification.h5')
    model_text_class.predict(padded)
    score = np.amax(model_text_class.predict(padded))
    classs = np.where(model_text_class.predict(padded) == np.amax(model_text_class.predict(padded)))[1][0]
    return score, classs

#Dictionary of Lbales
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

# Global Variables
text_fr = ''
text_en = ''

# Main Function
def main():

    global text_fr
    global text_en
    
    st.markdown("<style>h1{color: red;}</style>", unsafe_allow_html=True)
    st.markdown('# <div align="center">StAI Safe</div>', unsafe_allow_html=True)
    st.markdown(
        '## <div align="center">**Surveillance**</div>',
        unsafe_allow_html=True,
    )
    st.markdown("##  Utilisateur")

    pat_med = st.selectbox('', ("Patient", "Médecin"))

    if pat_med == "Patient":
        st.markdown("##  M. Dupont")
        st.markdown("###  Tension artérielle")

        tension_sys = st.number_input("Systolique", 60, 200, step=5, key="tension_sys")
        tension_dia = st.number_input("Diastolique", 50, 100, step=5, key="tension_dia")
        if st.button("Calculer Tension"):
            if (tension_sys < 80) or (tension_sys > 150) or (tension_dia < 60) or (tension_dia > 85):
                st.warning("Votre tension n'est pas normale, consulter votre médecin")
            elif (tension_sys > 170) or (tension_dia > 100):
                st.warning("Appeler Samu")
            else :
                st.success('Votre Tension est normale')

        if st.button("Décrivez votre symptôme"):
            result = from_mic()
            text_fr = result.text
            text_en = translatoor(text_fr)            
            scorr, clas = text_classify(text_en)
            label = dict_class[clas]
            st.success(text_fr)
            st.success(text_en)
            cursor.execute(
            "insert into dbo.Patient(NamePatient, Sentences, Label, Probability) values (?, ?, ?, ?)",
            "M. Dupont",f"{text_fr}", f"{label}", f"{scorr}")
            cnxn.commit()

    if pat_med == "Médecin":
        
        st.markdown("## Dr Watson")
        st.sidebar.title("INFOS DU PATIENT")

        cursor.execute(
            "SELECT TOP 1 NamePatient, Sentences, Label, Probability FROM dbo.Patient ORDER BY Id DESC" )
        row = cursor.fetchone()
        cnxn.commit()
        
        st.sidebar.write("Patient :", row[0])
        age = st.sidebar.number_input("Age", 45., 130., step=1., key="age")
        st.sidebar.write(" Historique d'hospitalisation :")
        number_medications = st.sidebar.number_input("Number Medications", 16., 75., step=1., key="number_medications")
        number_emergency  = st.sidebar.number_input("Number Emergency", 0., 54., step=1., key="number_emergency")
        number_impatient   = st.sidebar.number_input("Number impatient", 15., 19., step=1., key="number_impatient ")
        emergency_room  = st.sidebar.number_input("Emergency Room ", 0., 1., step=1., key="emergency_room")
        
        st.sidebar.write('')
        st.sidebar.write(" Infos sur l'état cardiaque :")

        ejection_fraction = st.sidebar.number_input(
            "Ejection Fraction : Pourcentage de sang quittant le cœur à chaque contraction", 20, 100, step=1, key="ejection_fraction"
        )
        serum_creatinine = st.sidebar.number_input(
            "Serum Creatinine : Niveau de créatinine sérique dans le sang", 2.0, 10.0, step=0.1, key="serum_creatinine"
        )
        time = st.sidebar.number_input("Time : Période de suivi (jours)", 10, 300, step=1, key="time")
        if st.button("Analyse"):
            #Model Cardiaque
            X_new = np.array([[ejection_fraction, serum_creatinine, time]])
            model_carqiaque = joblib.load("models/model_cardiaque.pkl")
            proba = model_carqiaque.predict_proba(X_new)[0][1]

            # Model Diabet
            scaler_diabete = joblib.load('models/scaler_diabete.pkl')
            model_diabete = joblib.load("models/risk_diabete.pkl")
            X_new1 = np.array([[age, number_medications, number_emergency, number_impatient, emergency_room]])
            X_new1_scal = scaler_diabete.transform(X_new1)
            probability = model_diabete.predict_proba(X_new1_scal)[0]

            st.write("Risque de réadmission hospitalière (diabète)")
            st.success("{} %".format(round(probability[1]*100, 2)))
            st.write("Risque cardiaque du patient")
            st.success("{} %".format(round(proba*100, 2)))
            st.write("Phrase détectée")
            st.success(row[1])
    
            st.write('Symptôme déduit')
            st.success(row[2])

            if ((proba > 0.8) and (probability[1]>0.8)) or (row[2]=='Heart hurts'):
                st.warning("Attention risque de crise carqiaque imminent")

            if ((proba > 0.8) and (probability[0]>0.8)) or (row[2]=='Heart hurts'):
                st.warning("Attention le patient présente un risque important de crise cardiaque")
       
if __name__ == "__main__":
    main()

