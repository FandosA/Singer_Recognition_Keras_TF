import pickle
import librosa
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}
    
    print('Extracting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []
        
        for i in range(0, wav.shape[0] - config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep = config.nfeat,
                     nfilt = config.nfilt, nfft = config.nfft)
            x = (x - config.min) / (config.max - config.min)
            
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis = 2)
            elif config.mode == 'feedforward':
                x = np.expand_dims(x, axis = 2)
            y_hat = model.predict(x.T)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
            
        fn_prob[fn] = np.mean(y_prob, axis = 0).flatten()
        
    return y_true, y_pred, fn_prob


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window = int(rate/10), min_periods = 1, center = True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


df = pd.read_csv('singers_test.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('wavfiles_ToPredict/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

df.reset_index(inplace = True)

for f in tqdm(df.fname):
    signal, rate = librosa.load('toPredict/'+f, sr = 16000)
    mask = envelope(signal, rate, 0.0005)
    wavfile.write(filename = 'clean_test/'+f, rate = rate, data = signal[mask])
    
    
# choose the model to test: 'conv.p', 'time.p' or 'feedforward.p'
model_to_test = 'feedforward.p'
    
df = pd.read_csv('singers_test.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles', model_to_test)

with open (p_path, 'rb') as handle:
    config = pickle.load(handle)
    
model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_predictions('clean_test')
acc_score = accuracy_score (y_true = y_true, y_pred = y_pred)

y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    for c, p in zip(classes, y_prob):
        df.at[i, c] = p
        
y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

if config.mode == 'conv':
    df.to_csv('Convolutional_Predictions.csv', index = False)
elif config.mode == 'time':
    df.to_csv('Recurrent_Predictions.csv', index = False)
elif config.mode == 'feedforward':
    df.to_csv('Feedforward_Predictions.csv', index = False)
