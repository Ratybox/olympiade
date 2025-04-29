import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
import librosa
from scipy.stats import variation
import warnings

# Suppression des avertissements
warnings.filterwarnings('ignore')

def extract_voice_features(audio_path):
    # Charger le fichier audio
    y, sr = librosa.load(audio_path, sr=None)

    # 1. Fréquence fondamentale (MDVP:Fo)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitches_nonzero = pitches[magnitudes > 0.1]
    pitch = np.mean(pitches_nonzero) if len(pitches_nonzero) > 0 else 0

    # 2. Fréquence maximale (MDVP:Fhi)
    max_pitch = np.max(pitches_nonzero) if len(pitches_nonzero) > 0 else 0

    # 3. Fréquence minimale (MDVP:Flo)
    min_pitch = np.min(pitches_nonzero) if len(pitches_nonzero) > 0 else 0

    # 4. Jitter (MDVP:Jitter) - variation de la fréquence fondamentale
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    f0_nonzero = f0[f0 > 0]
    jitter = variation(f0_nonzero) if len(f0_nonzero) > 0 else 0

    # 5. Jitter absolu (MDVP:Jitter(Abs))
    abs_jitter = np.mean(np.abs(np.diff(f0_nonzero))) if len(f0_nonzero) > 1 else 0

    # 6. Perturbation moyenne relative (MDVP:RAP)
    rap = np.mean(np.abs(np.diff(f0_nonzero) / f0_nonzero[:-1])) if len(f0_nonzero) > 1 else 0

    # 7. Quotient de perturbation de période de pitch (MDVP:PPQ)
    ppq = np.mean(np.abs(np.diff(f0_nonzero) / (f0_nonzero[:-1] + f0_nonzero[1:]) / 2)) if len(f0_nonzero) > 1 else 0

    # 8. Différence des différences de périodes (Jitter:DDP)
    ddp = np.mean(np.abs(np.diff(np.diff(f0_nonzero)))) if len(f0_nonzero) > 2 else 0

    # Extraire les MFCC pour features supplémentaires
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_vars = np.var(mfccs, axis=1)

    # Créer un dictionnaire des caractéristiques extraites
    features = {
        "MDVP:Fo(Hz)": pitch,
        "MDVP:Fhi(Hz)": max_pitch,
        "MDVP:Flo(Hz)": min_pitch,
        "MDVP:Jitter(%)": jitter,
        "MDVP:Jitter(Abs)": abs_jitter,
        "MDVP:RAP": rap,
        "MDVP:PPQ": ppq,
        "Jitter:DDP": ddp
    }

    # Ajout des features de Shimmer (variation d'amplitude)
    features["MDVP:Shimmer"] = mfcc_vars[0] if len(mfcc_vars) > 0 else 0
    features["MDVP:Shimmer(dB)"] = mfcc_vars[1] if len(mfcc_vars) > 1 else 0
    features["Shimmer:APQ3"] = mfcc_vars[2] if len(mfcc_vars) > 2 else 0
    features["Shimmer:APQ5"] = mfcc_vars[3] if len(mfcc_vars) > 3 else 0
    features["MDVP:APQ"] = mfcc_vars[4] if len(mfcc_vars) > 4 else 0
    features["Shimmer:DDA"] = mfcc_vars[5] if len(mfcc_vars) > 5 else 0

    # NHR, HNR (ratio harmoniques/bruit)
    features["NHR"] = np.mean(np.abs(y))
    features["HNR"] = librosa.feature.rms(y=y)[0].mean()

    # Autres mesures (RPDE, DFA, etc.)
    zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
    features["RPDE"] = np.mean(zero_crossings)
    features["DFA"] = np.std(y)
    features["spread1"] = np.percentile(mfccs.flatten(), 25)
    features["spread2"] = np.percentile(mfccs.flatten(), 75)
    features["D2"] = np.median(mfccs.flatten())
    features["PPE"] = np.max(np.abs(y))

    return features

def predict_parkinsons(audio_file, model_path='parkinsons_knn_model.pkl'):
    """
    Prédit si un fichier audio contient des marqueurs de la maladie de Parkinson
    
    Args:
        audio_file: Fichier audio à analyser
        model_path: Chemin du modèle sauvegardé
        
    Returns:
        Prédiction (1=Parkinson, 0=Sain) et score de confiance
    """
    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé à {model_path}. Veuillez vérifier le chemin du fichier.")
    
    # Charger le modèle
    try:
        model_data = joblib.load(model_path)
        knn_model = model_data['model']
        scaler = model_data['scaler']
        pca = model_data['pca']
        feature_names = model_data['feature_names']
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du modèle: {str(e)}")
    
    # Extraire les features audio avec la nouvelle fonction spécialisée
    audio_features = extract_voice_features(audio_file)
    
    # Créer un DataFrame avec les mêmes colonnes que celui utilisé pour l'entraînement
    features_df = pd.DataFrame([audio_features])
    
    # S'assurer que toutes les colonnes du modèle sont présentes
    for col in feature_names:
        if col not in features_df.columns:
            features_df[col] = 0
    
    # Réorganiser les colonnes comme pendant l'entraînement
    features_df = features_df[feature_names]
    
    # Normaliser
    normalized_features = scaler.transform(features_df)
    
    # Appliquer PCA
    pca_features = pca.transform(normalized_features)
    
    # Prédire
    prediction = knn_model.predict(pca_features)[0]
    
    # Calculer un score de confiance (distance aux voisins)
    distances, _ = knn_model.kneighbors(pca_features)
    confidence = 1 - (np.mean(distances) / 10)  # Normaliser sur [0,1]
    confidence = max(0, min(1, confidence))  # Assurer que c'est entre 0 et 1
    
    return prediction, confidence
