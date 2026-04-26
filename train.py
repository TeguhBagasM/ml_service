import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from app.preprocessing import preprocess_batch

def load_and_extract_features(csv_path: str):
    """
    Load dataset, preprocessing, dan ekstrak fitur TF-IDF.
    
    Returns:
        X_tfidf: Matrix TF-IDF
        y: Labels
        vectorizer: TfidfVectorizer yang sudah fit
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset loaded: {len(df)} baris")
    print(f"[INFO] Distribusi label:\n{df['status'].value_counts()}")
    
    # Validasi kolom
    assert 'judul' in df.columns, "Kolom 'judul' tidak ditemukan!"
    assert 'status' in df.columns, "Kolom 'status' tidak ditemukan!"
    
    # Hapus baris kosong
    df = df.dropna(subset=['judul', 'status'])
    
    # Preprocessing
    print("[INFO] Melakukan preprocessing teks...")
    df['judul_clean'] = preprocess_batch(df['judul'].tolist())
    
    # Tampilkan sampel hasil preprocessing
    print("\n[SAMPLE] Hasil preprocessing:")
    print(df[['judul', 'judul_clean', 'status']].head(5).to_string())
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=500,       # Ambil 500 fitur terpenting
        ngram_range=(1, 2),     # Unigram + bigram
        min_df=1,               # Minimum dokumen yang mengandung term
        sublinear_tf=True       # Gunakan log(TF) untuk normalisasi
    )
    
    X = df['judul_clean'].values
    y = df['status'].values
    
    X_tfidf = vectorizer.fit_transform(X)
    print(f"\n[INFO] Shape TF-IDF matrix: {X_tfidf.shape}")
    
    # Simpan vectorizer
    os.makedirs('model', exist_ok=True)
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    print("[INFO] Vectorizer disimpan ke model/vectorizer.pkl")
    
    return X_tfidf, y, vectorizer