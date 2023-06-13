import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import librosa #Python package for music & audio files
import librosa.display


from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from pickle import load

def get_model():
    convertor = LabelEncoder()
    class_list = ('blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')
    y = convertor.fit_transform(class_list)
    style = convertor.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    scaler = load(open('scaler.pkl', 'rb'))
    model = keras.models.load_model('mgen')
    return model, scaler, style

def get_meta(y, sr, scaler):
    #y, sr = librosa.load(p_t)
    length = 66149
    chroma_stft_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    chroma_stft_var = np.var(librosa.feature.chroma_stft(y=y, sr=sr))

    rms_mean = np.mean(librosa.feature.rms(y=y))
    rms_var = np.var(librosa.feature.rms(y=y))

    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_centroid_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))

    spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_bandwidth_var = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rolloff_var = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))

    zero_crossing_rate_mean = np.mean(librosa.feature.zero_crossing_rate(y))
    zero_crossing_rate_var = np.var(librosa.feature.zero_crossing_rate(y))

    harmony_mean = np.mean(librosa.effects.harmonic(y))
    harmony_var = np.var(librosa.effects.harmonic(y))

    perceptr_mean = np.mean(librosa.effects.percussive(y))
    perceptr_var = np.var(librosa.effects.percussive(y))

    tempo = librosa.feature.tempo(y=y, sr=sr)

    to_cl = {'length': length, 'chroma_stft_mean': chroma_stft_mean, 'chroma_stft_var': chroma_stft_var,
             'rms_mean': rms_mean, 'rms_var': rms_var, 'spectral_centroid_mean': spectral_centroid_mean,
             'spectral_centroid_var': spectral_centroid_var, 'spectral_bandwidth_mean': spectral_bandwidth_mean,
             'spectral_bandwidth_var': spectral_bandwidth_var, 'rolloff_mean': rolloff_mean, 'rolloff_var': rolloff_var,
             'zero_crossing_rate_mean': zero_crossing_rate_mean, 'zero_crossing_rate_var': zero_crossing_rate_var,
             'harmony_mean': harmony_mean, 'harmony_var': harmony_var, 'perceptr_mean': perceptr_mean,
             'perceptr_var': perceptr_var, 'tempo': tempo[0]}

    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    i = 1
    for e in mfcc:
        s_m = 'mfcc' + str(i) + '_mean'
        s_v = 'mfcc' + str(i) + '_var'

        to_cl[s_m] = np.mean(e)
        to_cl[s_v] = np.var(e)
        i += 1
    to_mod = scaler.transform(np.array(list(to_cl.values())).reshape(1, 58))
    return to_mod

def get_style_(y, sr, model, scaler, style):
    y = np.array(y)
    return style[np.argmax(model(get_meta(y, sr, scaler)))]

