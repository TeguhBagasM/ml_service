from flask import Flask, request, jsonify
import joblib
import os
from app.preprocessing import preprocess_text
from app.rules import get_rejection_reason

app = Flask(__name__)

# ─── LOAD MODEL & VECTORIZER ─────────────────────────────────────────────────
MODEL_PATH      = 'model/model.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'

try:
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("[✓] Model dan vectorizer berhasil dimuat.")
except FileNotFoundError as e:
    print(f"[✗] File tidak ditemukan: {e}")
    print("[!] Jalankan 'python train.py' terlebih dahulu.")
    model = vectorizer = None


# ─── HELPER ──────────────────────────────────────────────────────────────────
def predict_judul(judul: str) -> dict:
    """
    Prediksi kelayakan judul skripsi.
    
    Returns:
        dict dengan keys: status, alasan, confidence
    """
    if model is None or vectorizer is None:
        return {
            "status": "ERROR",
            "alasan": "Model belum tersedia. Jalankan training terlebih dahulu.",
            "confidence": 0.0
        }
    
    # Preprocessing
    judul_clean = preprocess_text(judul)
    
    # Vectorize
    X = vectorizer.transform([judul_clean])
    
    # Predict
    label       = model.predict(X)[0]
    proba       = model.predict_proba(X)[0]
    confidence  = float(max(proba))
    
    status = "DITERIMA" if label == 1 else "DITOLAK"
    
    # Rule-based reason
    alasan = get_rejection_reason(judul, judul_clean)
    
    # Override: jika ML bilang DITOLAK tapi rules bilang OK → tetap DITOLAK
    # Jika ML bilang DITERIMA tapi ada violations → turunkan ke DITOLAK
    has_violation = alasan.startswith("Judul ditolak")
    
    if has_violation:
        status = "DITOLAK"
    
    return {
        "status":     status,
        "alasan":     alasan,
        "confidence": round(confidence * 100, 2)
    }


# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "Skripsi Title Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Klasifikasi judul skripsi"
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint utama prediksi."""
    # Validasi Content-Type
    if not request.is_json:
        return jsonify({
            "error": "Content-Type harus application/json"
        }), 400
    
    data = request.get_json()
    
    # Validasi input
    judul = data.get('judul', '').strip()
    if not judul:
        return jsonify({
            "error": "Field 'judul' tidak boleh kosong."
        }), 422
    
    if len(judul) < 5:
        return jsonify({
            "error": "Judul terlalu pendek."
        }), 422
    
    # Prediksi
    result = predict_judul(judul)
    
    return jsonify({
        "judul":      judul,
        "status":     result['status'],
        "alasan":     result['alasan'],
        "confidence": f"{result['confidence']}%"
    }), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)