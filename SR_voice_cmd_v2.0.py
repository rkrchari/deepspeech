import numpy as np
import pandas as pd
import speech_recognition as sr
import pyttsx3
from pyttsx3 import voice
import streamlit as st
import webbrowser as wb
import time
import os
import nltk
from keybert import KeyBERT
global statement2,statement, key_text1, key_text2
#statement2=["show me the offers for Iphone", "Show me the offers for Mobile Phone", "I would like to buy Iphone", "I would like to buy mobile phone" ]

from sentence_transformers import SentenceTransformer, util
sentence_model = SentenceTransformer("all-mpnet-base-v2")

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

os.chdir("C:/Users//RAVI KUMAR//demo//deepspeechnew")

df = pd.read_csv('sentences.csv')
#statement2 = df['names'].tolist()
statement2 = df.values.tolist()

kw_model = KeyBERT(model=sentence_model)
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', 'voices[0].id')

def there_exists(terms):
    for term in terms:
        if term in statement:
            return True

def speak(text):
    engine.say(text)
    engine.runAndWait()

def takeCommand():
    r=sr.Recognizer()
    with sr.Microphone() as source:
         print("Listening..")
         audio = r.listen(source)
         
         try:
             statement=r.recognize_google(audio, language='en_in')
             print(f"user said:{statement}/n")

         except Exception as e:
             print("Pardon me, please say that againg")
             return "None"
         return statement

def keyword_ext(text):
    global statement
    stop_words = 'english'
    #keywords = kw_model.extract_keywords(statement, keyphrase_ngram_range=(4,2), stop_words = None, use_maxsum=True,nr_candidates=10, top_n=3)
    keywords = str(statement)
    #print(f"keywords {keywords} are extracted")
    return keywords

#mean ppolin - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model output contain all embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cosimilarity(key_text, text_corp):
    global statement, key_text1, key_text2
    key_text=str(key_text)
    key_text1 = key_text
    
    text_corp=list(text_corp)
    
    text_corp_tuple = []
    text_corp_tuple = [a_tuple[1] for a_tuple in text_corp]

    embeddings1 = sentence_model.encode(key_text, convert_to_tensor=True)
    #print(embeddings1)

# Load model from Hugging face hub
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    encoded_input = tokenizer(text_corp_tuple, padding=True, truncation=True, return_tensors='pt')

#  Compute token embeddings

    with torch.no_grad():
         model_output = model(**encoded_input)

#  Perform pooling. In this case, max pooling
   
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    #print(sentence_embeddings)
    
    top_k =1
    cos_scores=util.cos_sim(embeddings1, sentence_embeddings)[0]
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
    print("Sentence:", key_text, "\n")
    print("Top", top_k, "most similar sentences in corpus:")
    for idx in top_results[0:top_k]:
        print(text_corp[idx][2], "(Score: %.4f)" % (cos_scores[idx]))
    key_text1=text_corp[idx][2]
    print(f'keywords {key_text1} cosine similarity')

    
    #key_text2 = kw_model.extract_keywords(key_text1, keyphrase_ngram_range=(3,4), stop_words = 'english', use_maxsum=True,nr_candidates=10, top_n=2)
    return key_text1

if __name__=='__main__':
    WAKE = "hello"
    print("START")
    while True:
        statement = ''
        statement = takeCommand().lower()

        if statement.count(WAKE) > 0:
            print("Welcome to voice help")
            speak("Please tell me how can I help you now")
            statement = ''
            statement = takeCommand().lower() 
            statement=keyword_ext(statement)
            pos_tagged_sent = nltk.pos_tag(nltk.tokenize.word_tokenize(statement)) 
            nouns = [tag[0] for tag in pos_tagged_sent if tag[1]=='NN']
            search_term = nouns
            statement2 = df.values.tolist()
            #statement2 = df['names'].tolist()
            print(statement2)
#            statement2=["show me the offers for Iphone", "Show me the offers for Mobile Phone", "I would like to buy Iphone", "I would like to buy mobile phone" ]
            statement=cosimilarity(statement, statement2)
            print(statement)

            if statement == 0:
                continue
            if "good bye" in statement or "ok bye" in statement or "stop" in statement:
                speak('Good bye')
                print('Good bye')
                break

            elif "offer" in statement:
                url = "https://www.verizon.com/deals"
                wb.open(url)
                speak("Here is the various offers I found on Verizon Shop")

            elif "black friday" in statement:
            #elif there_exists(["black friday"]) or "verizon" in statement:
                url = 'https://www.verizon.com/black-friday'
                wb.open(url)
                speak("Here is the black friday deals I found on Verizon Shop")

            elif "free cellphone" in statement:
            #elif there_exists(["free cellphones", "free smartphones", "free cell phones", "free smart phones"]) or "verizon" in statement:
                url = 'https://www.verizon.com/shop/online/free-cell-phones'
                wb.open(url)
                speak("Here is the free smartphone deal i found on Verizon Shop")

            elif "preowned" in statement:
            #elif there_exists(["certified smartphones", "pre owned smartphones"]) or "verizon" in statement:
                url = 'https://www.verizon.com/shop/online/certified-pre-owned'
                wb.open(url)
                speak("Here is the certified smartphone deal i found on Verizon Shop")

            elif "certified" in statement:
            #elif there_exists(["certified smartphones", "pre owned smartphones"]) or "verizon" in statement:
                url = 'https://www.verizon.com/shop/online/certified-pre-owned'
                wb.open(url)
                speak("Here is the certified smartphone deal i found on Verizon Shop")
            
            elif "prepaid" in statement:
            #elif there_exists(["certified smartphones", "pre owned smartphones"]) or "verizon" in statement:
                url = 'https://www.verizon.com/deals/prepaid'
                wb.open(url)
                speak("Here is the prepaid plan details i found on Verizon Shop")

            elif "payment" in statement:
            #elif there_exists(["mobile bill", "payments", "billings"]) or "verizon" in statement:
                url = 'https://www.verizon.com/support/billing-and-payments'
                wb.open(url)
                speak("Please follow the instructions to understand your bills and payments")

            elif there_exists(["shop", "buy"]) or "verizon" in statement:
                search_term = statement.split("locate")[-1]
                url = 'https://www.verizon.com/onesearch/search?q=' + search_term + "&src=wireless"
                wb.get().open(url)
                speak("Here is what I found for " + search_term + "on verizon shop")
                time.sleep(4) 

            elif "byod" in statement:
            #elif there_exists(["mobile bill", "payments", "billings"]) or "verizon" in statement:
                url = 'https://www.verizon.com/bring-your-own-device/'
                wb.open(url)
                speak("Here is the details found on bring your own device on verizon shop")

            elif "byod" in statement:
            #elif there_exists(["mobile bill", "payments", "billings"]) or "verizon" in statement:
                url = 'https://www.verizon.com/bring-your-own-device/'
                wb.open(url)
                speak("Here is the details found on bring your own device on verizon shop")

            elif "fiveg" in statement:
            #elif there_exists(["mobile bill", "payments", "billings"]) or "verizon" in statement:
                url = 'https://www.verizon.com/5g-home-internet/'
                wb.open(url)
                speak("Here is the details found on 5g Home Internet on verizon shop")     
 
            elif "fios" in statement:
            #elif there_exists(["mobile bill", "payments", "billings"]) or "verizon" in statement:
                url = 'https://www.verizon.com/deals/fios/'
                wb.open(url)
                speak("Here is the details found on fiber optic service on verizon shop")  

            elif there_exists(["locate"]) or "verizon" in statement:
                search_term = statement.split("locate")[-1]
                url = 'https://www.google.com/maps/search' + search_term
                wb.get().open(url)
                speak("Here is what I found for " + search_term)
                time.sleep(4) 
            elif 'time' in statement:
                strTime = datetime.now().strftime('%H:%M')
                speak(f"the time is  + {time}")
            elif "make a note" in statement:
                speak("Please dictate me the note")
                note = takeCommand()
                datetime = datetime,now()
                ts1 = datetime.strftime("%d-%b-%Y/%H:%M:%S")
                file = open('note.txt', 'w')
                speak("Do you need to include date and time")
                snfm = takeCommand()
                if 'yes' in snfm:
                    strTime = datetime.strftime("%d-%b-%Y/%H:%M:%S")
                    file.write(strTime)
                    file.write(":-")
                    file.write(note)
                else:
                    file.write(note)
                file.close()

                speak("your not is saved as note.txt")
            time.sleep(3) 