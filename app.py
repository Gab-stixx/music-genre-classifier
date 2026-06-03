import streamlit as st
import pickle
import numpy as np
import librosa
import os

# Load model and encoder
@st.cache_resource
def load_model():
    with open('genre_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

def extract_audio_features(audio_file):
    """Extract audio features using librosa"""
    # Load audio file
    y, sr = librosa.load(audio_file, sr=22050)
    
    # Duration
    duration_ms = librosa.get_duration(y=y, sr=sr) * 1000
    
    # Tempo (simplified)
    try:
        tempo = librosa.feature.tempogram(y=y, sr=sr).mean()
        tempo = min(tempo * 100, 250)  # Scale and cap
    except:
        tempo = 120  # Default
    
    # Energy
    rms = librosa.feature.rms(y=y)[0]
    energy = np.mean(rms)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # Features dict
    features = {
        'danceability': min(energy * 1.5, 1.0),
        'energy': min(energy * 2, 1.0),
        'key': 0,
        'loudness': 20 * np.log10(np.mean(rms) + 1e-9) - 20,
        'mode': 0,
        'speechiness': np.mean(zero_crossing_rate),
        'acousticness': 0.5,
        'instrumentalness': 0.5,
        'liveness': np.std(rms),
        'valence': 0.5,
        'tempo': tempo,
        'time_signature': 4,
        'popularity': 50,
        'duration_ms': duration_ms,
        'explicit': 0
    }
    
    return features

model, le = load_model()

st.title("🎵 Music Genre Classifier")
st.write("Upload an audio file to predict its genre!")

# File upload
uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'ogg', 'flac'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open('temp_audio', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("🔄 Extracting audio features...")
    
    try:
        # Extract features
        features_dict = extract_audio_features('temp_audio')
        
        # Prepare feature array in correct order
        feature_order = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                        'time_signature', 'popularity', 'duration_ms', 'explicit']
        
        features = np.array([[features_dict[f] for f in feature_order]])
        
        # Predict
        prediction = model.predict(features)
        genre = le.inverse_transform(prediction)[0]
        
        st.success(f"🎼 Predicted Genre: **{genre.upper()}**")
        
        # Show extracted features
        with st.expander("View extracted features"):
            for key, value in features_dict.items():
                st.write(f"**{key}:** {value:.3f}")
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
    
    # Clean up
    os.remove('temp_audio')