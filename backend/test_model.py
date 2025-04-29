#!/usr/bin/env python
"""
Script de test pour tester le modèle Parkinson préentraîné
"""
import os
import sys
from api.utils import predict_parkinsons

def main():
    # Définir le chemin du modèle
    model_path = os.path.join(os.path.dirname(__file__), 'api', 'parkinsons_knn_model.pkl')
    
    # Informations sur le modèle
    if os.path.exists(model_path):
        print(f"Modèle trouvé: {model_path}")
        print(f"Taille du modèle: {os.path.getsize(model_path) / 1024:.2f} KB")
    else:
        print(f"ERREUR: Modèle non trouvé à {model_path}")
        return
    
    # Tester avec un fichier audio
    test_file = input("\nEntrez le chemin d'un fichier audio à tester (ou appuyez sur Entrée pour quitter): ")
    if test_file and os.path.exists(test_file):
        print(f"Analyse du fichier {test_file}...")
        
        try:
            # Prévoir avec le modèle existant
            prediction, confidence = predict_parkinsons(test_file, model_path)
            
            # Afficher les résultats
            result = "Parkinson détecté" if prediction == 1 else "Sain"
            print(f"\nRésultat: {result}")
            print(f"Confiance: {confidence:.2f}")
        except Exception as e:
            print(f"ERREUR lors de la prédiction: {str(e)}")
    else:
        print("Aucun fichier valide fourni. Test terminé.")
    
    print("Script terminé.")

if __name__ == "__main__":
    main() 