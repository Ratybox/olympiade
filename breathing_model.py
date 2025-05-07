import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam

# Path to the dataset
data_path = './data'

# Load data
def load_data(data_path):
    labels = []
    features = []
    for label in os.listdir(data_path):
        class_path = os.path.join(data_path, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=512, hop_length=256)
                    mfcc_mean = np.mean(mfcc.T, axis=0)
                    features.append(mfcc_mean)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return np.array(features), np.array(labels)

# Load features and labels
features, labels = load_data(data_path)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=0.2, random_state=42)

# Reshape features for 1D CNN input AFTER train/test split
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # (samples, time_steps, features)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the model - switching to 1D convolution which is better for this data shape
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(40, 1)))
model.add(MaxPooling1D(2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))

# Save using the SavedModel format
model.save('audio_classification_model.keras')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

