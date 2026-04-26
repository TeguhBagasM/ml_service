import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from app.preprocessing import preprocess_batch


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load CSV dengan auto-detect separator dan kolom."""

    # Coba berbagai separator
    for sep in ['\t', ',', ';', '|']:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            df.columns = df.columns.str.strip().str.lower()

            # Debug: tampilkan info
            print(f"[DEBUG] Separator: {repr(sep)}")
            print(f"[DEBUG] Kolom ditemukan: {df.columns.tolist()}")
            print(f"[DEBUG] Shape: {df.shape}")
            print(f"[DEBUG] Preview:\n{df.head(3)}\n")

            # Cek apakah kolom target ada
            if 'judul' in df.columns and 'status' in df.columns:
                print("[✓] Kolom 'judul' dan 'status' ditemukan!")
                return df
            else:
                print(f"[✗] Kolom tidak cocok, coba separator lain...\n")
        except Exception as e:
            print(f"[✗] Error dengan sep={repr(sep)}: {e}")

    # Jika semua gagal, tampilkan isi raw file
    print("\n[ERROR] Tidak bisa detect kolom. Isi raw file:")
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            print(f"  Baris {i}: {repr(line)}")
            if i >= 4:
                break

    raise ValueError("Tidak bisa load dataset. Cek format CSV kamu.")


def train_and_evaluate():

    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset.csv')

    # ── 1. LOAD ───────────────────────────────────────────────────
    df = load_dataset(csv_path)
    df = df.dropna(subset=['judul', 'status'])

    print(f"Total data  : {len(df)}")
    print(f"Distribusi  :\n{df['status'].value_counts()}\n")

    # Cek minimum data
    min_per_class = df['status'].value_counts().min()
    if min_per_class < 5:
        raise ValueError(
            f"Data terlalu sedikit! Minimal 5 per kelas, kamu punya {min_per_class}. "
            "Tambah data dulu di dataset.csv"
        )

    # ── 2. PREPROCESSING ─────────────────────────────────────────
    df['judul_clean'] = preprocess_batch(df['judul'].tolist())

    X = df['judul_clean'].values
    y = df['status'].astype(int).values

    # ── 3. SPLIT ──────────────────────────────────────────────────
    # Kurangi test_size jika data sedikit
    n_samples   = len(df)
    test_size   = 0.2 if n_samples >= 20 else 0.15
    cv_folds    = min(5, min_per_class)  # CV fold menyesuaikan jumlah data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | CV folds: {cv_folds}")

    # ── 4. TF-IDF ─────────────────────────────────────────────────
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # ── 5. MODELS ─────────────────────────────────────────────────
    models = {
        'Naive Bayes': MultinomialNB(alpha=0.5),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=42, n_jobs=-1
        )
    }

    # ── 6. TRAINING & EVALUASI ────────────────────────────────────
    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"  MODEL: {name}")
        print(f"{'='*50}")

        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(classification_report(y_test, y_pred, target_names=['DITOLAK', 'DITERIMA']))
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        cv = cross_val_score(model, X_train_tfidf, y_train, cv=cv_folds, scoring='accuracy')
        print(f"CV Score : {cv.mean():.4f} ± {cv.std():.4f}")

        results[name] = {'model': model, 'accuracy': acc, 'cv_mean': cv.mean()}

    # ── 7. PILIH TERBAIK ──────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]['cv_mean'])
    best_model = results[best_name]['model']

    print(f"\n{'='*50}")
    print(f"  BEST MODEL : {best_name}")
    print(f"  Accuracy   : {results[best_name]['accuracy']:.4f}")
    print(f"  CV Score   : {results[best_name]['cv_mean']:.4f}")
    print(f"{'='*50}")

    # ── 8. SIMPAN ─────────────────────────────────────────────────
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))

    print("\n[✓] model/model.pkl tersimpan")
    print("[✓] model/vectorizer.pkl tersimpan")


if __name__ == '__main__':
    train_and_evaluate()