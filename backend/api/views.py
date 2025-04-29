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
import traceback

# Import de nos fonctions d'IA
from .utils import predict_parkinsons

# Chemin du modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'parkinsons_knn_model.pkl')

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
        
        # Prédire avec notre modèle
        try:
            prediction, confidence = predict_parkinsons(temp_path, MODEL_PATH)
            
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
        
        return JsonResponse(response_data, status=200 if response_data['status'] == 'success' else 500)
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e),
            'stack_trace': traceback.format_exc()
        }, status=500)

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
        
        # Vérifier si le modèle est chargé
        model_exists = os.path.exists(MODEL_PATH)
        model_size = os.path.getsize(MODEL_PATH) if model_exists else 0
        
        return JsonResponse({
            'status': 'success',
            'message': 'API en bonne santé',
            'model_loaded': model_exists,
            'model_path': MODEL_PATH,
            'model_size': model_size,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
