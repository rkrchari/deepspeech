from flask import Flask, render_template, send_from_directory, Response, jsonify, request, make_response
from pathlib import Path
#from audioemotion import analyzeTone
import numpy as np
import wave
import base64
import json
import os
#from audioemotion import writeBase64AudioString
#from flask_api import response
from capture_int import capture_and_save
from camera_int import Camera
from tone_org import speechEmotionRecognition_org
import argparse
import cv2

camera = Camera()

#camera = cv2.VideoCapture(0)
camera.run()
#model_sub_dir = os.path.join('C://Users//RAVI KUMAR//demo//deepspeechnew//emo_age_gender//flask//MODEL_CNN_LSTM.hdf5')
#SER = speechEmotionRecognition_org(model_sub_dir)

# Prediction emotion in voice at each time stamp

step = 1 # in sec
sample_rate = 16000 # in khz


app = Flask(__name__)
# app.config["SECRET_KEY"] = "secret!"

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering or Chrome Frame,
    and also to cache the rendered page for 10 minutes
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers["Cache-Control"] = "public, max-age=0"
    return r


@app.route("/")
def entrypoint():
    return render_template("index.html")


@app.route("/capture")
def capture():
    im = camera.get_frame(bytes=False)
    capture_and_save(im)
    return render_template("send_to_init.html")


@app.route("/images/last")
def last_image():
    p = Path("images/last.png")
    if p.exists():
        r = "last.png"
        gender = camera.gender
        age = camera.age
        emotion = camera.emotion
        sentiment = camera.sentiment
        timestamp = camera.timestamp
    else:
        print("No last")
        r = "not_found.jpeg"
    return send_from_directory("images",r)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

def get_result(camera):
    emotion = camera.emotion
    gender = camera.gender
    age = camera.age
    prediction = camera.prediction
    sentiment = camera.sentiment
    timestamp = camera.timestamp
    #return emotion, gender, age, prediction, timestamp
    return emotion, gender, age, prediction, sentiment, timestamp



@app.route("/stream")
def stream_page():
    gender = "..."
    age = "..."
    emotion = "..."
    prediction = "..."
    sentiment = "..."
    timestamp = "..."
    return render_template("stream.html", gender = gender, age = age, emotion = emotion, prediction=prediction,  sentiment=sentiment, timestamp = timestamp)
    #return render_template("stream.html", gender = gender, age = age, emotion = emotion, prediction=prediction, timestamp = timestamp)

@app.route("/get_prediction")
def get_prediction():
    emotion, gender, age, prediction, sentiment, timestamp = get_result(camera)
    #emotion, gender, age, prediction, timestamp = get_result(camera)
 #   rec_sub_dir = os.path.join("C://Users//RAVI KUMAR//demo//deepspeechnew//emo_age_gender//flask//resonate_24.wav")
 #   rec_sub_dir1 = request.files["rec_sub_dir"]
#    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir1, chunk_step=step*sample_rate)
#    audio_emotion = get_result1(speech)
    return render_template("stream.html", emotion=emotion, gender=gender, age=age, prediction=prediction,sentiment=sentiment, timestamp = timestamp)
    #return render_template("stream.html", emotion=emotion, gender=gender, age=age, prediction=prediction,timestamp = timestamp)

#@app.route("/get_prediction_tone", methods=['POST'])
#def get_prediction_tone():
#    #emotion, gender, age = get_result(camera)
#    rec_sub_dir = os.path.join("C://Users//RAVI KUMAR//demo//deepspeechnew//emo_age_gender//flask//resonate_24.wav")
#    rec_sub_dir1 = request.files["rec_sub_dir"]
#    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir1, chunk_step=step*sample_rate)
#    return render_template("stream.html",audio_emotion=emotions)


@app.route("/video_feed")
def video_feed():
    return Response(gen(camera),
        mimetype="multipart/x-mixed-replace; boundary=frame")



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--port',type=int,default=5000, help="Running port")
    parser.add_argument("-H","--host",type=str,default='0.0.0.0', help="Address to broadcast")
    args = parser.parse_args()
    app.run(host=args.host,port=args.port)



