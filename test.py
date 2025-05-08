import os
import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from PIL import Image
import torch
from torchvision import transforms

from model import CoughVIDModel


audio_path = sys.argv[1]

def extract_features(audio_path):
    audio_clip, sample_rate = librosa.load(audio_path)
    
    spec = librosa.stft(audio_clip)
    spec_mag, _ = librosa.magphase(spec)
    mel_spec = librosa.feature.melspectrogram(S=spec_mag, sr=sample_rate)
    log_spec = librosa.amplitude_to_db(mel_spec, ref=np.min)

    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('tight')
    ax.axis('off')
    img = librosa.display.specshow(log_spec, sr=sample_rate)
    img_path = os.path.splitext(audio_path)[0] + '.png'
    fig.savefig(img_path)
    plt.close(fig)

    mfcc = librosa.feature.mfcc(y=audio_clip, sr=sample_rate)  # Fixed this line
    mfcc = preprocessing.scale(mfcc, axis=1)

    return img_path, mfcc


img_path, mfcc = extract_features(audio_path)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open(img_path)
img = img.convert('RGB')
img = transform(img)
img = img.unsqueeze(0)

mfcc_padded = np.zeros((20, 500))
mfcc_padded[:mfcc.shape[0], :mfcc.shape[1]] = mfcc[:, :500]
mfcc = torch.Tensor(mfcc_padded)
mfcc = mfcc.unsqueeze(0)

model = torch.load('model.h5', map_location='cpu')

out = model(img, mfcc)

pred = torch.argmax(out, 1)

if pred == 0.0:
    print('Healthy')
else:
    print('COVID-19')
    