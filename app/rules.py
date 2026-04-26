import re
from typing import List


# ─── KONFIGURASI ─────────────────────────────────────────────────────────────

KATA_METODE = {
    'algoritma', 'metode', 'klasifikasi', 'prediksi', 'analisis',
    'deteksi', 'segmentasi', 'clustering', 'regresi', 'forecasting',
    'identifikasi', 'komparasi', 'perbandingan', 'evaluasi', 'optimasi',
    'rekomendasi', 'pengenalan', 'ekstraksi', 'implementasi',
    # nama algoritma spesifik
    'naive', 'bayes', 'knn', 'svm', 'lstm', 'cnn', 'rnn',
    'random', 'forest', 'decision', 'tree', 'neural', 'network',
    'deep', 'learning', 'machine', 'k-means', 'kmeans',
    'xgboost', 'gradient', 'boosting', 'linear', 'logistic',
}

KATA_OBJEK_PENELITIAN = {
    # domain data
    'data', 'teks', 'gambar', 'citra', 'suara', 'video', 'sinyal',
    # domain aplikasi
    'penyakit', 'cuaca', 'harga', 'saham', 'kredit', 'spam',
    'pelanggan', 'mahasiswa', 'siswa', 'pasien', 'karyawan',
    'produk', 'berita', 'ulasan', 'sentimen', 'emosi',
    'wajah', 'sidik', 'tanaman', 'pertanian', 'curah',
    'kelulusan', 'nilai', 'prestasi', 'penjualan', 'transaksi',
    'jaringan', 'intrusi', 'hoaks', 'disinformasi', 'energi',
    'listrik', 'suhu', 'kelembaban', 'gempa', 'banjir',
}

# Frasa yang membuat judul TERLALU UMUM (harus muncul sebagai keseluruhan judul)
FRASA_TERLALU_UMUM = [
    'sistem informasi',
    'aplikasi web',
    'aplikasi mobile',
    'website',
    'sistem akademik',
    'toko online',
    'aplikasi android',
    'sistem manajemen',
]

MIN_KATA = 5    # minimal kata dalam judul asli
MAX_KATA = 25   # maksimal kata dalam judul asli

# Singkatan: huruf besar semua, 2+ karakter, BUKAN kata umum
KATA_BUKAN_SINGKATAN = {
    'web', 'php', 'api', 'sql', 'css', 'html', 'iot', 'ai', 'ml',
    'ui', 'ux', 'erp', 'crm', 'cms', 'url', 'http', 'json', 'xml',
    # singkatan institusi/umum yang wajar
    'pt', 'cv', 'smk', 'sma', 'smp', 'sd', 'rs', 'rsud', 'pln',
    'bpjs', 'ktp', 'nik', 'nip', 'npm',
    # akronim teknologi yang dikenal
    'lstm', 'cnn', 'rnn', 'svm', 'knn', 'ann', 'mlp',
    'naive', 'id',
}


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def _words_original(judul: str) -> List[str]:
    """Tokenisasi judul original (lowercase)."""
    return judul.lower().split()


def _has_metode(judul: str) -> bool:
    """Apakah judul mengandung kata metode/algoritma."""
    words = set(_words_original(judul))
    # Cek juga substring (misal 'k-nearest' mengandung 'nearest')
    judul_lower = judul.lower()
    for kata in KATA_METODE:
        if kata in judul_lower:
            return True
    return False


def _has_objek(judul: str) -> bool:
    """Apakah judul mengandung objek penelitian spesifik."""
    judul_lower = judul.lower()
    for kata in KATA_OBJEK_PENELITIAN:
        if kata in judul_lower:
            return True
    return False


def _is_terlalu_umum(judul: str) -> bool:
    """
    Judul terlalu umum jika:
    - Sama persis dengan frasa umum, ATAU
    - Hanya terdiri dari frasa umum + sedikit kata tambahan (≤ 3 kata)
    """
    judul_lower = judul.lower().strip()
    kata = judul_lower.split()

    for frasa in FRASA_TERLALU_UMUM:
        frasa_kata = frasa.split()
        # Judul persis sama
        if judul_lower == frasa:
            return True
        # Judul dimulai dengan frasa umum dan total kata sedikit
        if judul_lower.startswith(frasa) and len(kata) <= len(frasa_kata) + 3:
            return True

    return False


def _is_terlalu_pendek(judul: str) -> bool:
    """Judul terlalu pendek (< MIN_KATA kata)."""
    return len(judul.split()) < MIN_KATA


def _is_terlalu_panjang(judul: str) -> bool:
    """Judul terlalu panjang (> MAX_KATA kata)."""
    return len(judul.split()) > MAX_KATA


def _has_singkatan_berlebihan(judul: str) -> bool:
    """
    Singkatan berlebihan = ada 3+ token yang:
    - Semua huruf kapital
    - Panjang 2-6 karakter
    - BUKAN kata yang dikenal (teknologi, institusi, dll)
    - BUKAN nama algoritma
    """
    tokens = judul.split()
    singkatan_asing = []

    for token in tokens:
        # Bersihkan tanda baca
        token_clean = re.sub(r'[^A-Za-z]', '', token)
        if (
            len(token_clean) >= 2
            and len(token_clean) <= 6
            and token_clean == token_clean.upper()        # semua kapital
            and token_clean.lower() not in KATA_BUKAN_SINGKATAN
            and not token_clean.isdigit()
        ):
            singkatan_asing.append(token_clean)

    return len(singkatan_asing) >= 3


# ─── EVALUASI RULES ──────────────────────────────────────────────────────────

def get_rejection_reason(judul_original: str, judul_clean: str) -> str:
    """
    Evaluasi semua rules dan kembalikan alasan.

    Returns:
        String alasan penolakan, atau pesan diterima.
    """
    violations = []

    # Rule 1: Terlalu pendek
    if _is_terlalu_pendek(judul_original):
        violations.append(
            f"Judul terlalu pendek (kurang dari {MIN_KATA} kata). "
            "Tambahkan metode dan objek penelitian."
        )

    # Rule 2: Terlalu panjang
    if _is_terlalu_panjang(judul_original):
        violations.append(
            f"Judul terlalu panjang (lebih dari {MAX_KATA} kata). Sederhanakan."
        )

    # Rule 3: Terlalu umum
    if _is_terlalu_umum(judul_original):
        violations.append(
            "Judul terlalu umum dan tidak spesifik. "
            "Tambahkan metode, objek penelitian, dan konteks yang jelas."
        )

    # Rule 4: Tidak ada metode (hanya jika judul tidak terlalu umum/pendek)
    if not _is_terlalu_pendek(judul_original) and not _has_metode(judul_original):
        violations.append(
            "Judul tidak mencantumkan metode atau algoritma yang digunakan. "
            "Contoh: 'Menggunakan Algoritma Decision Tree' atau 'Berbasis Machine Learning'."
        )

    # Rule 5: Tidak ada objek penelitian
    if not _is_terlalu_pendek(judul_original) and not _has_objek(judul_original):
        violations.append(
            "Judul tidak mencantumkan objek atau domain penelitian yang spesifik. "
            "Contoh: data penjualan, penyakit diabetes, ulasan pelanggan."
        )

    # Rule 6: Singkatan berlebihan (hanya jika bukan karena terlalu pendek)
    if not _is_terlalu_pendek(judul_original) and _has_singkatan_berlebihan(judul_original):
        violations.append(
            "Terlalu banyak singkatan yang tidak umum. "
            "Tulis kepanjangan singkatan agar judul lebih mudah dipahami."
        )

    # ── Output ───────────────────────────────────────────────────
    if violations:
        alasan_list = "\n".join(f"• {v}" for v in violations)
        return f"Judul ditolak karena:\n{alasan_list}"

    return (
        "Judul sudah sesuai. Mengandung metode yang jelas, "
        "objek penelitian spesifik, dan panjang yang tepat."
    )