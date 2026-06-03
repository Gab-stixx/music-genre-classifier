# 🎵 Music Genre Classifier

A machine learning web app that predicts music genres from audio files using XGBoost classification.

## Features

- Upload audio files (MP3, WAV, OGG, FLAC)
- Automatically extracts audio features
- Predicts music genre using trained ML model
- Simple, user-friendly interface

## Genres Supported

Pop, Rock, Hip-Hop, Country, R&B, Jazz, Classical, Electronic, Indie, Metal, Folk, Blues, Reggae, Alternative, Dance

## Tech Stack

- **Python** - Backend
- **Scikit-learn** - ML preprocessing
- **XGBoost** - Classification model
- **Librosa** - Audio feature extraction
- **Streamlit** - Web interface

## How to Run Locally

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/music-genre-classifier.git
cd music-genre-classifier

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## Model Performance

- Accuracy: 52% on 15 music genres
- Trained on 114,000 Spotify tracks
- Uses XGBoost with hyperparameter tuning

## Usage

1. Upload an audio file
2. App extracts features automatically
3. Model predicts the genre
4. View results and extracted features

## Author

Gabriel Komolafe

## License

MIT