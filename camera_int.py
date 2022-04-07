import cv2
import os
import pickle
import numpy as np
import threading
import wave, time
import face_recognition
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
import tone_org
import array
from array import array
from array import *
import pyaudio
import scipy.io.wavfile as wav
from sklearn import preprocessing
from sklearn.preprocessing import normalize
#This is for audio
from sys import byteorder
import pydub
from pydub import AudioSegment
from pydub.silence import split_on_silence
import base64
from sklearn.preprocessing import normalize
from nltk.tokenize import sent_tokenize, word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from tone_org import speechEmotionRecognition_org
import speech_recognition as sr
speech = ''


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORDS_SECONDS = 3
chunk_step=16000
step = 1 # in sec
sample_rate = 16000 # in khz

WAVE_OUTPUT_FILENAME = 'file.wav'
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("* recording")


thread = None
os.chdir('C://Users//RAVI KUMAR//demo//deepspeechnew//emo_age_gender//flask')
emo_model = load_model("CNNVer1.53.hdf5")
gen_model = load_model("gender_model.39.hdf5")
age_model = load_model("age_model_cnn.42.hdf5")
model_sub_dir = os.path.join('MODEL_CNN_LSTM.hdf5')
model1 = load_model('sentiment_keras.hdf5')
with open('C:/Users/RAVI KUMAR/demo/deepspeechnew/emo_age_gender/flask/tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle) 

genders = ('female', 'male')
ages = ('0-15', '16-25', '26-35', '36-45', '>46')
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
speech = ''

def record(WAVE_OUTPUT_FILENAME):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
#    print("* recording")

    frames=[]

    for i in range(0, int(RATE/ CHUNK * RECORDS_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) 

#    print("*done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wave_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(p.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()
    import base64 
    #enc = base64.b64encode(open("file.wav", "rb").read())
    #prediction = speechEmotionRecognition_org.predict_emotion_from_file(WAVE_OUTPUT_FILENAME) 


    SER = speechEmotionRecognition_org(model_sub_dir)
    rec_sub_dir = os.path.join(WAVE_OUTPUT_FILENAME)
    
    prediction_1, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)
    major_emotion = max(set(prediction_1), key=prediction_1.count)
    return major_emotion, timestamp



def speechToText(WAVE_OUTPUT_FILENAME):
    speech = ''
    r = sr.Recognizer()
    hellow=sr.AudioFile(WAVE_OUTPUT_FILENAME)
    with hellow as source:
        audio = r.record(source)
    try:
        speech = r.recognize_google(audio)
        print("Text: "+speech)
    except Exception as e:
        print("Exception: "+str(e))
    return speech

def sentiment1(speech):
    print(speech)
    tw = load_tokenizer.texts_to_sequences([speech])
    tw = pad_sequences(tw, maxlen=200)
    sentiment1 = int(model1.predict(tw).round().item())
    if sentiment1 == 0:
       sentiment = "negative"
    elif sentiment1 == 1:
       sentiment = "positive"
    return sentiment


def predict_gender_age(detected_face):
    #preprocess image
    detected_face_gen = tf.compat.v1.image.resize(detected_face, [64,64])#resize to 192x192
    face_gen = tf.keras.preprocessing.image.img_to_array(detected_face_gen)
    face_gen = face_gen / 255.
    face_gen /= float(face_gen.max())
    #face_gen = np.expand_dims(face_gen, axis = 0)
    face_gen = np.reshape(face_gen.flatten(), (1,64,64,3))
    

    #predict gender
    results = gen_model.predict(face_gen)
    gen_predictions = results[0]

    
    age_predictions = age_model.predict(face_gen)

    age_predictions =  round(age_predictions[0,0])
    
    #max_index_gen = np.argmax(gen_predictions[0])
    #max_index_age = np.argmax(age_predictions[0])

    gender = gen_predictions
    gender = "F" if gender > 0.5 else "M"
    age = format(int(age_predictions))
    return gender, age
    return age


def predict_emotion(detected_face):
    #preprocess image
    detected_face_emo = tf.image.rgb_to_grayscale(detected_face) #convert image to grayscale
    detected_face_emo = tf.compat.v1.image.resize(detected_face_emo, [48,48]) #resize image
    face_emo = tf.keras.preprocessing.image.img_to_array(detected_face_emo)
    face_emo = face_emo /255.0
    #face_emo = face_emo.astype(np.float32)


    face_emo /= float(face_emo.max())
    face_emo = np.reshape(face_emo.flatten(), (1,48,48,1))
    #face_emo = face_emo /255.0

    #face_emo = np.expand_dims(face_emo, axis = 0)
    #face_emo = tf.reshape(face_emo, [1,48,48,1])


    #face_emo = face_emo/255.0 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
    emo_predictions = emo_model.predict(face_emo) #store probabilities of 7 expressions

    #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
    max_index = np.argmax(emo_predictions[0])
    emotion = emotions[max_index]
    return emotion


class Camera:
    def __init__(self,fps=20,video_source=0):
        self.fps = fps
        self.video_source = video_source
        self.camera = cv2.VideoCapture(self.video_source)
        # We want a max of 5s history to be stored, thats 5s*fps
        self.max_frames = 1 * self.fps
        self.frames = []
        self.gender = 'No result'
        self.age = 'No result'
        self.emotion = 'No result'
        self.prediction = 'No Result'
        self.timestamp = 'No Result'
        self.sentiment = 'No Result'
        self.isrunning = False
    def run(self):
        global thread
        if thread is None:
            thread = threading.Thread(target=self._capture_loop)
            print("Starting thread...")
            thread.start()
            self.isrunning = True

    def _capture_loop(self):
        dt = 1/self.fps
        print("Observing...")
        while self.isrunning:

            try:
                v,im = self.camera.read()
                faces = face_recognition.face_locations(im)
                for face_location in faces:
                  # Print the location of each face in this image
                    top, right, bottom, left = face_location
                    detected_face = im[int(top-35):int(bottom+35), int(left-35):int(right+35)]
                    self.emotion = predict_emotion(detected_face)
                    cv2.rectangle(im, (left,top), (right, bottom), (0,255,0), 2)
                    if self.gender == 'No result' and self.age == 'No result':
                       self.gender, self.age = predict_gender_age(detected_face)
                       self.age = predict_gender_age(detected_face)
                       #speech = speechToText(WAVE_OUTPUT_FILENAME)
                       #self.sentiment = sentiment1(speech)
                    
                if v:
                    if len(self.frames)==self.max_frames:
                       self.frames = self.frames[1:]
                    self.frames.append(im)
                    #if self.prediction == 'No result' and self.timestamp == 'No result':
                    self.prediction, self.timestamp = record(WAVE_OUTPUT_FILENAME)

                    speech = speechToText(WAVE_OUTPUT_FILENAME)
                    self.sentiment = sentiment1(speech)
                #time.sleep(dt)

            except:
                #return 0
                if self.prediction == 'No result' and self.timestamp == 'No result':
                   self.prediction, self.timestamp = record(WAVE_OUTPUT_FILENAME)

                #self.prediction, self.timestamp = record(WAVE_OUTPUT_FILENAME)
                   speech = speechToText(WAVE_OUTPUT_FILENAME)
                   self.sentiment = sentiment1(speech)



    def stop(self):
        self.isrunning = False
    def get_frame(self, bytes=True):
        if len(self.frames)>0:
            if bytes:
                img = cv2.imencode('.png',self.frames[-1])[1].tobytes()
            else:
                img = self.frames[-1]
        else:
            with open("images/not_found.jpeg","rb") as f:
                img = f.read()
        return img



        