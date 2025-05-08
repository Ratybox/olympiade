from django.shortcuts import render
from rest_framework.response import Response
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
import json
import os
from datetime import datetime
import numpy as np
from scipy.io import wavfile
import librosa
import joblib
import traceback
import base64
from PIL import Image
from io import BytesIO
import tensorflow as tf
from torchvision import transforms
import tempfile
# Set matplotlib to use a non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which doesn't require a GUI
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
from torch.serialization import add_safe_globals, safe_globals

# Import de nos fonctions d'IA
from .utils import predict_parkinsons, predict_updrs_score

import torch
import torch.nn as nn


class CoughVIDModel(nn.Module):
  def __init__(self, base_model, mfcc_in_shape):
    super().__init__()
    self.base_model = base_model
    self.mfcc_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(mfcc_in_shape[0]*mfcc_in_shape[1], 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 1000),
        nn.ReLU(),
        nn.Dropout(0.2),
    )
    self.classifier = nn.Sequential(
        nn.Linear(2000, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 2),
    )
  
  def forward(self, img, mfcc):
    out_1 = self.base_model(img)
    out_2 = self.mfcc_model(mfcc)
    out_merged = torch.cat([out_1, out_2], dim=1)
    out = self.classifier(out_merged)
    return out

# Chemins des modèles
KNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'parkinsons_knn_model.pkl')
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'parkinsons_rf_model.pkl')
COUGH_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cough.h5')

# Add CoughVIDModel to safe globals
add_safe_globals([CoughVIDModel])

@api_view(['POST'])
@parser_classes([MultiPartParser])
def detect_cough(request):
    if 'audio' not in request.FILES:
        return Response({"error": "Audio file missing"}, status=status.HTTP_400_BAD_REQUEST)

    audio_file = request.FILES['audio']

    # Save audio file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        for chunk in audio_file.chunks():
            tmp_audio.write(chunk)
        tmp_audio_path = tmp_audio.name

    try:
        # Extract features
        audio_clip, sample_rate = librosa.load(tmp_audio_path)

        spec = librosa.stft(audio_clip)
        spec_mag, _ = librosa.magphase(spec)
        mel_spec = librosa.feature.melspectrogram(S=spec_mag, sr=sample_rate)
        log_spec = librosa.amplitude_to_db(mel_spec, ref=np.min)

        # Save spectrogram to temp image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis('tight')
            ax.axis('off')
            librosa.display.specshow(log_spec, sr=sample_rate)
            fig.savefig(tmp_img.name)
            plt.close(fig)
            img_path = tmp_img.name

        # Load and transform spectrogram image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # Process MFCC
        mfcc = librosa.feature.mfcc(y=audio_clip, sr=sample_rate)
        mfcc = preprocessing.scale(mfcc, axis=1)
        mfcc_padded = np.zeros((20, 500))
        mfcc_padded[:mfcc.shape[0], :mfcc.shape[1]] = mfcc[:, :500]
        mfcc_tensor = torch.Tensor(mfcc_padded).unsqueeze(0)

        # Load model
        try:
            # First create a base model (using a simple CNN as an example)
            base_model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 112 * 112, 1000)
            )
            
            # Initialize the model with the base model
            model = CoughVIDModel(base_model, (20, 500))
            
            # Try loading just the state dict first
            try:
                state_dict = torch.load(COUGH_MODEL_PATH, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading state dict: {str(e)}")
                # If state dict loading fails, try loading the full model
                try:
                    # Create a temporary module to hold the model class
                    import types
                    temp_module = types.ModuleType('model')
                    temp_module.CoughVIDModel = CoughVIDModel
                    import sys
                    sys.modules['model'] = temp_module
                    
                    with safe_globals([CoughVIDModel]):
                        model = torch.load(COUGH_MODEL_PATH, map_location='cpu', weights_only=False)
                except Exception as e2:
                    print(f"Error loading full model: {str(e2)}")
                    raise
            
            model.eval()
        except Exception as e:
            print(f"Error in model loading process: {str(e)}")
            raise

        with torch.no_grad():
            output = model(img_tensor, mfcc_tensor)
            prediction = torch.argmax(output, dim=1).item()

        result = "Healthy" if prediction == 0 else "COVID-19"
        print('result: ', result)
        return Response({"prediction": prediction}, status=status.HTTP_200_OK)

    except Exception as e:
        print(e)
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        os.remove(tmp_audio_path)
        if os.path.exists(img_path):
            os.remove(img_path)


@csrf_exempt
@api_view(['POST'])
def parkinson(request):
    try:
        base64_image = request.data.get('image')
        if not base64_image:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Decode the base64 image
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data)).convert('RGB')  # Ensure RGB
        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 224, 224, 3)

        parkinson_model_path = os.path.join(os.path.dirname(__file__), 'parkinson_spiral.h5')
        print(parkinson_model_path)
        # Load your Xception model
        model = tf.keras.models.load_model(parkinson_model_path)  # or .keras if saved that way

        # Predict
        prediction = model.predict(image_array)[0][0]
        predicted_class = int(prediction > 0.5)
        print({
            "predicted_class": predicted_class,
            "confidence": float(prediction)
        })

        return Response({
            "predicted_class": predicted_class,
            "confidence": float(prediction)
        }, status=status.HTTP_200_OK)

    except Exception as e:
        print(e)
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@csrf_exempt
@require_http_methods(["POST"])
def upload(request):
    try:
        # Vérifier si un fichier audio est présent
        if 'audio' not in request.FILES:
            return JsonResponse({
                'status': 'error',
                'message': 'Aucun fichier audio fourni'
            }, status=400)

        audio_file = request.FILES['audio']
        timestamp = request.POST.get('timestamp', datetime.now().isoformat())

        # Valider le type de fichier
        valid_extensions = ['.m4a', '.mp3', '.wav', '.mp4', '.3gp']
        ext = os.path.splitext(audio_file.name)[1].lower()
        if ext not in valid_extensions:
            return JsonResponse({
                'status': 'error',
                'message': 'Type de fichier invalide'
            }, status=400)

        # Valider la taille du fichier (limite de 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if audio_file.size > max_size:
            return JsonResponse({
                'status': 'error',
                'message': 'Fichier trop volumineux'
            }, status=400)

        # Créer le nom de fichier avec timestamp
        filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        file_path = os.path.join('recordings', filename)

        # Créer le dossier s'il n'existe pas
        os.makedirs('recordings', exist_ok=True)

        # Sauvegarder le fichier
        with open(file_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        return JsonResponse({
            'status': 'success',
            'filename': filename,
            'file_path': file_path,
            'file_size': audio_file.size,
            'file_type': audio_file.content_type,
            'timestamp': timestamp
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    try:
        # Vérifier si un fichier audio est présent
        if 'audio' not in request.FILES:
            return JsonResponse({
                'status': 'error',
                'message': 'Aucun fichier audio fourni'
            }, status=400)
            
        audio_file = request.FILES['audio']
        
        # Enregistrer temporairement le fichier
        temp_path = os.path.join('temp', f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        os.makedirs('temp', exist_ok=True)
        
        with open(temp_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)
        
        # Prédire avec notre modèle KNN (classification binaire)
        try:
            prediction, confidence = predict_parkinsons(temp_path, KNN_MODEL_PATH)
            
            # Interpréter la prédiction
            result = "Parkinson détecté" if prediction == 1 else "Sain"
            
            response_data = {
                'status': 'success',
                'prediction': int(prediction),
                'result': result,
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            # En cas d'erreur de prédiction, capturer l'erreur
            error_msg = f"Erreur lors de la prédiction: {str(e)}"
            stack_trace = traceback.format_exc()
            response_data = {
                'status': 'error',
                'message': error_msg,
                'stack_trace': stack_trace
            }
        finally:
            # Nettoyer le fichier temporaire
            try:
                os.remove(temp_path)
            except:
                pass
        
        return JsonResponse(response_data, status=200 )
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e),
            'stack_trace': traceback.format_exc()
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def predict_updrs(request):
    """Prédire le score UPDRS moteur pour évaluer la sévérité de Parkinson"""
    try:
        # Vérifier si un fichier audio est présent
        if 'audio' not in request.FILES:
            return JsonResponse({
                'status': 'error',
                'message': 'Aucun fichier audio fourni'
            }, status=400)
            
        audio_file = request.FILES['audio']
        
        # Enregistrer temporairement le fichier
        temp_path = os.path.join('temp', f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        os.makedirs('temp', exist_ok=True)
        
        with open(temp_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)
        
        # Prédire avec notre modèle RandomForest (régression)
        try:
            result = predict_updrs_score(temp_path, RF_MODEL_PATH)
            
            # Ajouter des informations supplémentaires à la réponse
            response_data = {
                'status': 'success',
                'updrs_score': round(result['updrs_score'], 2),
                'severity': result['severity'],
                'severity_code': result['severity_code'],
                'explanation': get_severity_explanation(result['severity_code']),
                'top_features': result['top_features'],
                'model_info': result['model_info'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            # En cas d'erreur de prédiction, capturer l'erreur
            error_msg = f"Erreur lors de la prédiction UPDRS: {str(e)}"
            stack_trace = traceback.format_exc()
            response_data = {
                'status': 'error',
                'message': error_msg,
                'stack_trace': stack_trace
            }
        finally:
            # Nettoyer le fichier temporaire
            try:
                os.remove(temp_path)
            except:
                pass
        
        return JsonResponse(response_data, status=200 if response_data['status'] == 'success' else 500)
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e),
            'stack_trace': traceback.format_exc()
        }, status=500)

def get_severity_explanation(severity_code):
    """Retourne une explication détaillée du niveau de sévérité"""
    explanations = {
        0: "Aucun symptôme significatif ou symptômes très légers. Les activités quotidiennes ne sont pas affectées.",
        1: "Symptômes légers. Quelques difficultés dans les mouvements fins, mais les activités quotidiennes restent largement indépendantes.",
        2: "Symptômes modérés. Difficultés notables dans la motricité, la parole peut être affectée. Une assistance occasionnelle peut être nécessaire.",
        3: "Symptômes sévères. Mobilité significativement réduite, difficultés importantes dans la parole et les activités quotidiennes. Une assistance régulière est recommandée."
    }
    return explanations.get(severity_code, "Niveau de sévérité non reconnu.")

@require_http_methods(["GET"])
def recordings(request):
    try:
        recordings_dir = "recordings"
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)
            
        files = [f for f in os.listdir(recordings_dir) if any(f.endswith(ext) for ext in ['.wav', '.mp3', '.m4a'])]
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

@require_http_methods(["GET"])
def health_check(request):
    """Vérifie l'état de l'API"""
    try:
        # Vérifier si le dossier recordings existe
        os.makedirs('recordings', exist_ok=True)
        
        # Vérifier si les modèles sont chargés
        knn_model_exists = os.path.exists(KNN_MODEL_PATH)
        knn_model_size = os.path.getsize(KNN_MODEL_PATH) if knn_model_exists else 0
        
        rf_model_exists = os.path.exists(RF_MODEL_PATH)
        rf_model_size = os.path.getsize(RF_MODEL_PATH) if rf_model_exists else 0
        
        return JsonResponse({
            'status': 'success',
            'message': 'API en bonne santé',
            'models': {
                'knn': {
                    'loaded': knn_model_exists,
                    'path': KNN_MODEL_PATH,
                    'size': knn_model_size
                },
                'rf': {
                    'loaded': rf_model_exists,
                    'path': RF_MODEL_PATH,
                    'size': rf_model_size
                }
            },
            'timestamp': datetime.now().isoformat(),
            'version': '1.1.0'
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
