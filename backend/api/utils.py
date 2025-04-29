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

    print("features: ", features)

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

def extract_updrs_features(audio_path):
    """
    Extrait les caractéristiques vocales optimisées pour la prédiction du score UPDRS
    """
    print(f"Analyse du fichier: {audio_path}")
    
    # Vérification de l'existence du fichier
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Fichier audio introuvable: {audio_path}")
    
    try:
        # Chargement avec paramètres optimisés
        y, sr = librosa.load(audio_path, sr=44100, duration=3.0)  # Normalisation de la durée
    except Exception as e:
        raise RuntimeError(f"Erreur de chargement audio: {str(e)}")
    
    # Prétraitement
    y = librosa.effects.preemphasis(y, coef=0.97)  # Compensation de l'atténuation haute fréquence
    y = librosa.util.normalize(y)  # Normalisation du volume
    
    # Caractéristiques utilisées par le modèle
    selected_features = [
        'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 
        'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
    ]
    
    # Initialisation
    features = {col: 0.0 for col in selected_features}
    
    # Extraction du pitch et calcul du Jitter
    f0, voiced_flag, _ = librosa.pyin(
        y, 
        fmin=80,  # Fréquence minimale réaliste pour la voix humaine
        fmax=300, # Fréquence maximale réaliste
        frame_length=4096
    )
    
    # Calculs du Jitter
    f0_clean = f0[voiced_flag & ~np.isnan(f0)]
    if len(f0_clean) > 1:
        diffs = np.diff(f0_clean)
        features["Jitter(%)"] = (np.std(diffs) / np.mean(f0_clean)) * 100
        features["Jitter(Abs)"] = np.mean(np.abs(diffs))
        
        # Jitter:RAP (perturbation relative moyenne)
        rap_vals = []
        for i in range(1, len(f0_clean)-1):
            rap_vals.append(abs(f0_clean[i] - (f0_clean[i-1] + f0_clean[i] + f0_clean[i+1])/3))
        features["Jitter:RAP"] = np.mean(rap_vals) / np.mean(f0_clean) if np.mean(f0_clean) != 0 else 0
        
        # Jitter:PPQ5 (quotient de perturbation de 5 points)
        ppq5_vals = []
        for i in range(2, len(f0_clean)-2):
            ppq5_vals.append(abs(f0_clean[i] - (f0_clean[i-2] + f0_clean[i-1] + f0_clean[i] + f0_clean[i+1] + f0_clean[i+2])/5))
        features["Jitter:PPQ5"] = np.mean(ppq5_vals) / np.mean(f0_clean) if np.mean(f0_clean) != 0 else 0
    
    # Calcul du Shimmer avec fenêtrage dynamique
    amplitude = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    shimmer_vals = []
    for i in range(1, amplitude.shape[1]-1):
        shimmer_vals.append(np.abs(amplitude[:,i].mean() - 
                              (amplitude[:,i-1].mean() + amplitude[:,i+1].mean())/2))
    features["Shimmer"] = np.mean(shimmer_vals) if shimmer_vals else 0.0
    
    # Shimmer en dB
    if shimmer_vals:
        features["Shimmer(dB)"] = 20 * np.log10(np.mean(shimmer_vals)) if np.mean(shimmer_vals) > 0 else 0
    
    # Shimmer APQ3 (quotient de perturbation d'amplitude de 3 points)
    shimmer_apq3 = []
    for i in range(1, amplitude.shape[1]-1):
        val = abs(amplitude[:,i].mean() - (amplitude[:,i-1].mean() + amplitude[:,i].mean() + amplitude[:,i+1].mean())/3)
        shimmer_apq3.append(val)
    features["Shimmer:APQ3"] = np.mean(shimmer_apq3) if shimmer_apq3 else 0
    
    # Shimmer APQ5 (quotient de perturbation d'amplitude de 5 points)
    shimmer_apq5 = []
    for i in range(2, amplitude.shape[1]-2):
        window = [amplitude[:,j].mean() for j in range(i-2, i+3)]
        val = abs(amplitude[:,i].mean() - sum(window)/5)
        shimmer_apq5.append(val)
    features["Shimmer:APQ5"] = np.mean(shimmer_apq5) if shimmer_apq5 else 0
    
    # Calcul optimisé du HNR
    D = librosa.stft(y)
    S, _ = librosa.magphase(D)
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    
    features["HNR"] = 20 * np.log10(np.mean(harmonic**2) / (np.mean(percussive**2) + 1e-7))
    features["NHR"] = 1.0 / (features["HNR"] + 1e-7) if features["HNR"] > 0 else 10.0
    
    # Calcul dynamique du RPDE
    rpde_vals = []
    for i in range(100, len(y), 100):
        segment = y[i-100:i]
        zcr = librosa.feature.zero_crossing_rate(segment)
        rpde_vals.append(zcr.mean())
    features["RPDE"] = np.std(rpde_vals) if rpde_vals else 0
    
    # DFA (Detrended Fluctuation Analysis)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["DFA"] = np.std(spectral_centroid)
    
    # PPE (Pitch Period Entropy)
    if len(f0_clean) > 0:
        hist, bin_edges = np.histogram(f0_clean, bins=10, density=True)
        hist = hist[hist > 0]  # Exclure les valeurs nulles pour éviter log(0)
        entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
        features["PPE"] = entropy
    
    return features

def predict_updrs_score(audio_file, model_path='parkinsons_rf_model.pkl'):
    """
    Prédit le score UPDRS moteur à partir d'un fichier audio avec le nouveau modèle optimisé
    
    Args:
        audio_file: Fichier audio à analyser
        model_path: Chemin du modèle RandomForest sauvegardé
        
    Returns:
        Score UPDRS prédit, niveau de sévérité et données supplémentaires
    """
    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle RF non trouvé à {model_path}. Veuillez vérifier le chemin du fichier.")
    
    # Charger le modèle
    try:
        model_data = joblib.load(model_path)
        rf_model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']  # Modifié pour utiliser juste les noms des caractéristiques (sans PCA)
        metrics = model_data.get('metrics', {})
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du modèle RF: {str(e)}")
    
    # Extraire les features audio avec la nouvelle fonction spécialisée pour UPDRS
    audio_features = extract_updrs_features(audio_file)
    
    print("features: ", audio_features)
    
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
    
    # Prédire le score UPDRS (directement sans PCA)
    updrs_score = float(rf_model.predict(normalized_features)[0])
    
    # Déterminer le niveau de sévérité
    severity = "Normal"
    severity_code = 0
    
    if updrs_score > 40:
        severity = "Sévère"
        severity_code = 3
    elif updrs_score > 25:
        severity = "Modéré"
        severity_code = 2
    elif updrs_score > 15:
        severity = "Léger"
        severity_code = 1
    
    # Préparer des informations complémentaires
    feature_importances = {}
    if hasattr(rf_model, 'feature_importances_'):
        # Récupérer les caractéristiques les plus importantes
        importances = rf_model.feature_importances_
        # Associer chaque importance à son nom de caractéristique
        feature_imp = [(feature_names[i], float(imp)) for i, imp in enumerate(importances)]
        # Trier par importance décroissante et prendre les top 5
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        feature_importances = {name: value for name, value in feature_imp[:5]}
    
    return {
        'updrs_score': updrs_score,
        'severity': severity,
        'severity_code': severity_code,
        'model_info': {
            'mse': metrics.get('mse', 0),
            'rmse': metrics.get('rmse', 0),
            'r2': metrics.get('r2', 0)
        },
        'top_features': feature_importances,
        'raw_features': {
            'jitter': features_df['Jitter(%)'].values[0] if 'Jitter(%)' in features_df.columns else 0,
            'shimmer': features_df['Shimmer'].values[0] if 'Shimmer' in features_df.columns else 0,
            'hnr': features_df['HNR'].values[0] if 'HNR' in features_df.columns else 0
        }
    }
