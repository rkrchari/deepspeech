import json
import os
import numpy as np
import wave
import pandas as pd
import librosa
from scipy.stats import zscore
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import base64
from scipy.io import wavfile
import scipy.io.wavfile
from scipy.io.wavfile import read
import wave
import soundfile
import warnings
warnings.filterwarnings("ignore")


class speechEmotionRecognition_org:

    def __init__(self, subdir_model=None):
        if subdir_model is not None:
            self._model = self.load_model()
            self._model.load_weights(subdir_model)
  
        self._emotion = {0: 'Fear', 1: 'Angry', 2: 'Neutral', 3: 'Happy', 4: 'Sad', 5:'Surprise'}

    ## Mel Spectogram computation
    def mel_spectogram(self, y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
   
    # compute spectogram
        mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) **2

     # compute mel spectogram
        mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)

     # compute mel spectogram
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
      
        return np.asarray(mel_spect)  

    ## Audio Framing

    def frame(self, y, win_step=64, win_size = 128):

        # Number of frames
        nb_frames = 1 + int((y.shape[2] - win_size ) / win_step )

        # Framming
        frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size)).astype(np.float16)
        for t in range(nb_frames):
            frames[:, t,:,:] = np.copy(y[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float16)
        return frames


     ## RNN model

    def load_model(self):
         model = load_model('C://Users//RAVI KUMAR//demo//deepspeechnew//emo_age_gender//flask//MODEL_CNN_LSTM.hdf5')

         return model
     ## Predict speech emotion over time from an audio file
    def predict_emotion_from_file(self, filename, chunk_step=16000, chunk_size=49100, predict_proba=False, sample_rate = 16000):
     # Read Audio file
         y, sr = librosa.core.load(filename, sr=sample_rate, offset=0.5)

     # Padding or truncated signal

         if len(y) < chunk_size:
             y_padded = np.zeros(chunk_size)
             y_padded[:len(y)] = y
             y = y_padded

     # Split audio signals into chunks              

         chunks = self.frame(y.reshape(1, 1,-1), chunk_step, chunk_size)

     # Reshape chunks
   
         #chunks = chunks.reshape(chunks.shape[1], chunks.shape[-1])    
         chunks = chunks.reshape(chunks.shape[1],chunks.shape[-1] )    
                        
     # Z- Normalization
         y = np.asarray(list(map(zscore, chunks)))

     # Compute mel spectogram
         mel_spect = np.asarray(list(map(self.mel_spectogram, y)))
         
     # Time distributed framing
         mel_spect_ts = self.frame(mel_spect)

     # Build X for modelling
         X = mel_spect_ts.reshape(mel_spect_ts.shape[0],
                                  mel_spect_ts.shape[1],
                                  mel_spect_ts.shape[2],
                                  mel_spect_ts.shape[3],1)

     # Predict emotion

         if predict_proba is True:
             predict = self._model.predict(X)
         else:
             predict = np.argmax(self._model.predict(X), axis = 1)
             predict = [self._emotion.get(emotion) for emotion in predict]

      # Clear Keras Session
         K.clear_session()

        # Predict Timestamp

         timestamp = np.concatenate([[chunk_size], np.ones((len(predict) - 1)) * chunk_step]).cumsum()
         timestamp = np.round(timestamp / sample_rate)

         return [predict, timestamp]


    def prediction_to_csv(self, predictions, filename, mode='w'):

     # write emotions in filename
        with open(filename, mode) as f:
            if mode == 'w':
                f.write("EMOTIONS"+'\n')
            for emotion in predictions:
                f.write(str(emotion)+'\n')
            f.close();
 
     # Split audio signals into chunks    


#def main():

#    model_sub_dir = os.path.join('C://Users//RAVI KUMAR//demo//deepspeechnew//emo_age_gender//flask//MODEL_CNN_LSTM.hdf5')
#    SER = speechEmotionRecognition_org(model_sub_dir)
#    rec_sub_dir = os.path.join(WAVE_OUTPUT_FILENAME)

# Prediction emotion in voice at each time stamp

    step = 1 # in sec
    sample_rate = 16000 # in khz

# Emotion with timestamp

#    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)
#    print(emotions)

## Export predicted emotion to .txt format
#    SER.prediction_to_csv(emotions, os.path.join("audio_emotions.txt"), mode='w')

# Get most common emotion
#    major_emotion = max(set(emotions), key=emotions.count)

# Calculate emotion distribution

#    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]


# Export emotion distribution to .csv format 

#    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
#    df.to_csv(os.path.join('audio_emotions_dist.txt'), sep=',')

#if __name__ == '__main__':
#    main()
































        