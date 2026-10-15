import librosa
import librosa.feature
import numpy as np


class AudioFeatureExtractor:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def extract(self, file_path):
        #ucitavanje zvuka
        y, _ = librosa.load(file_path, sr=self.sr)

        #provjera je li datoteka prazna
        if len(y) == 0:
            print("Upozorenje: Datoteka", file_path, "je prazna!")
            return np.zeros(14)

        #mel-spektrogram
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=2048,
            hop_length=512
        )

        #logaritamska skala
        log_S = librosa.power_to_db(S)

        #MFCC (13 koeficijenata)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        #zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        #spajanje znacajki (13 MFCC + 1 ZCR)
        features = np.hstack([mfcc_mean, zcr_mean])

        return features
