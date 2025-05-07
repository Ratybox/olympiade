from django.shortcuts import render
from rest_framework.response import Response
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework import status
from rest_framework.decorators import api_view
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

# Import de nos fonctions d'IA
from .utils import predict_parkinsons, predict_updrs_score

# Chemins des modèles
KNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'parkinsons_knn_model.pkl')
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'parkinsons_rf_model.pkl')


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
