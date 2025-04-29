import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import librosa
from scipy.stats import variation
import os

# Function to suppress warnings (Optional)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Function to extract features from the audio
def extract_voice_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # 1. Fundamental frequency (MDVP:Fo)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches)

    # 2. Maximum pitch (MDVP:Fhi)
    max_pitch = np.max(pitches)

    # 3. Minimum pitch (MDVP:Flo)
    min_pitch = np.min(pitches)

    # 4. Jitter (MDVP:Jitter)
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    jitter = variation(f0)

    # 5. Absolute jitter (MDVP:Jitter(Abs))
    abs_jitter = np.mean(np.abs(np.diff(f0)))

    # 6. Relative Average Perturbation (MDVP:RAP)
    rap = np.mean(np.abs(np.diff(f0) / f0[:-1]))

    # 7. Pitch Period Perturbation Quotient (MDVP:PPQ)
    ppq = np.mean(np.abs(np.diff(f0) / (f0[:-1] + f0[1:])))

    # 8. Difference of Differences of Periods (Jitter:DDP)
    ddp = np.mean(np.abs(np.diff(np.diff(f0))))

    # Create a dictionary of the extracted features
    features_of_the_sound = {
        "MDVP:Fo(Hz)": pitch,
        "MDVP:Fhi(Hz)": max_pitch,
        "MDVP:Flo(Hz)": min_pitch,
        "MDVP:Jitter(%)": jitter,
        "MDVP:Jitter(Abs)": abs_jitter,
        "MDVP:RAP": rap,
        "MDVP:PPQ": ppq,
        "Jitter:DDP": ddp
    }

    return features_of_the_sound


# Load data from your dataset (use for training the model)
data_path = r"C:\Users\USER\.cache\kagglehub\datasets\naveenkumar20bps1137\parkinsons-disease-detection\versions\1\parkinsons.data"
df = pd.read_csv(data_path, sep=',')  # Adjust `sep` if needed

# Prepare data for training (using 'status' as target variable)
df = df.drop(columns=['name'])
features = df.drop(columns=['status'])
target = df['status']

# Normalize features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features) 
normalized_features_df = pd.DataFrame(normalized_features, columns=features.columns)

# Apply PCA to retain 95% of the variance
pca = PCA(n_components=0.95)
principal_components = pca.fit_transform(normalized_features)

# Convert the principal components into a DataFrame for better interpretability
principal_components_df = pd.DataFrame(
    principal_components,
    columns=[f"PC{i+1}" for i in range(principal_components.shape[1])]
)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(principal_components_df, target, test_size=0.2, random_state=42)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Display model performance
print(f"Model Performance on Test Data:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")

# Example usage: Predict the label of an audio file
audio_path = "amir.wav"  # Replace with the path to your audio file
extracted_features = extract_voice_features(audio_path)
print("Sound features:")
print(extracted_features)

# Convert the extracted features to a DataFrame
extracted_features_df = pd.DataFrame([extracted_features])

# Align the extracted features with the training feature set
# Missing features will be filled with 0, ensuring column order consistency
aligned_features = pd.DataFrame(columns=features.columns)
aligned_features = pd.concat([aligned_features, extracted_features_df], ignore_index=True).fillna(0)

# Normalize only the features that exist in the extracted data
aligned_normalized_features = scaler.transform(aligned_features)

# Apply PCA to the normalized features
pca_aligned_features = pca.transform(aligned_normalized_features)

# Predict the label using the trained KNN model
predicted_label = knn.predict(pca_aligned_features)
print(f"Predicted Label for the given sound: {predicted_label[0]}")
