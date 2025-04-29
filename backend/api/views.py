from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import os
from datetime import datetime
import numpy as np
from scipy.io import wavfile
import librosa
import joblib

# Modèle de machine learning (à remplacer par votre modèle réel)
model = None

@csrf_exempt
@require_http_methods(["POST"])
def upload(request):
    try:
        audio_file = request.FILES['audio']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        
        # Sauvegarder le fichier
        with open(f"recordings/{filename}", 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)
                
        return JsonResponse({
            'status': 'success',
            'filename': filename
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)

@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    try:
        audio_file = request.FILES['audio']
        
        # Charger l'audio
        y, sr = librosa.load(audio_file, sr=None)
        
        # Extraire les caractéristiques
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Faire la prédiction (exemple simplifié)
        prediction = model.predict([mfccs_mean])[0] if model else 0
        
        return JsonResponse({
            'status': 'success',
            'prediction': float(prediction),
            'confidence': 0.95  # À remplacer par la vraie confiance du modèle
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)

@require_http_methods(["GET"])
def recordings(request):
    try:
        recordings_dir = "recordings"
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)
            
        files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
        return JsonResponse({
            'status': 'success',
            'recordings': files
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)

@require_http_methods(["GET"])
def recording_detail(request, recording_id):
    try:
        filename = f"recording_{recording_id}.wav"
        filepath = os.path.join("recordings", filename)
        
        if not os.path.exists(filepath):
            return JsonResponse({
                'status': 'error',
                'message': 'Recording not found'
            }, status=404)
            
        return JsonResponse({
            'status': 'success',
            'filename': filename,
            'size': os.path.getsize(filepath),
            'created_at': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)
