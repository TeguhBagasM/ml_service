"""
Rule-based reasoning untuk menghasilkan alasan DITERIMA/DITOLAK.
Rules mudah dikembangkan — cukup tambahkan di RULES list.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Rule:
    """Representasi satu aturan pengecekan."""
    name: str
    check: callable     # Fungsi yang menerima (judul_original, judul_clean) → bool
    reason: str         # Alasan jika rule terpenuhi → DITOLAK
    reject: bool = True # True = rule ini menyebabkan penolakan


# ─── KATA KUNCI DOMAIN ───────────────────────────────────────────────────────
KATA_METODE = {
    'algoritma', 'metode', 'klasifikasi', 'prediksi', 'analisis',
    'deteksi', 'sistem', 'implementasi', 'penerapan', 'optimasi',
    'segmentasi', 'clustering', 'regresi', 'forecasting', 'identifikasi',
    'komparasi', 'perbandingan', 'evaluasi', 'naive', 'bayes', 'knn',
    'svm', 'lstm', 'cnn', 'random', 'forest', 'decision', 'tree',
    'neural', 'network', 'deep', 'learning', 'machine'
}

KATA_OBJEK = {
    'data', 'teks', 'gambar', 'citra', 'suara', 'video', 'sensor',
    'penyakit', 'cuaca', 'harga', 'saham', 'kredit', 'pelanggan',
    'mahasiswa', 'siswa', 'pasien', 'produk', 'berita', 'ulasan',
    'sentimen', 'emosi', 'wajah', 'sidik', 'tanaman', 'pertanian'
}

JUDUL_TERLALU_UMUM = {
    'sistem informasi', 'aplikasi web', 'website', 'aplikasi mobile',
    'sistem', 'aplikasi', 'website sekolah', 'toko online'
}

JUDUL_TERLALU_PENDEK_KATA = 4   # Minimal 4 kata bermakna
JUDUL_TERLALU_PANJANG_KATA = 25 # Maksimal 25 kata


def _has_kata_metode(judul_original: str, judul_clean: str) -> bool:
    """Cek apakah judul mengandung kata metode/teknik."""
    words = set(judul_original.lower().split())
    return bool(words & KATA_METODE)


def _has_kata_objek(judul_original: str, judul_clean: str) -> bool:
    """Cek apakah judul mengandung objek/domain penelitian."""
    words = set(judul_original.lower().split())
    return bool(words & KATA_OBJEK)


def _is_terlalu_umum(judul_original: str, judul_clean: str) -> bool:
    """Cek apakah judul terlalu generik/umum."""
    judul_lower = judul_original.lower().strip()
    # Cek exact match atau judul hanya berisi frasa umum
    for frasa in JUDUL_TERLALU_UMUM:
        if judul_lower == frasa or judul_lower.startswith(frasa + ' ') \
                and len(judul_lower.split()) <= 4:
            return True
    return False


def _is_terlalu_pendek(judul_original: str, judul_clean: str) -> bool:
    """Cek apakah judul terlalu pendek."""
    kata_bersih = judul_clean.split()
    return len(kata_bersih) < JUDUL_TERLALU_PENDEK_KATA


def _is_terlalu_panjang(judul_original: str, judul_clean: str) -> bool:
    """Cek apakah judul terlalu panjang."""
    kata = judul_original.split()
    return len(kata) > JUDUL_TERLALU_PANJANG_KATA


def _has_singkatan_tidak_jelas(judul_original: str, judul_clean: str) -> bool:
    """Cek singkatan berlebihan (≥ 3 singkatan sekaligus)."""
    singkatan = re.findall(r'\b[A-Z]{2,}\b', judul_original)
    return len(singkatan) >= 3


# ─── DAFTAR RULES ────────────────────────────────────────────────────────────
RULES = [
    Rule(
        name="terlalu_pendek",
        check=_is_terlalu_pendek,
        reason="Judul terlalu pendek. Tambahkan objek penelitian dan metode yang digunakan."
    ),
    Rule(
        name="terlalu_panjang",
        check=_is_terlalu_panjang,
        reason=f"Judul terlalu panjang (lebih dari {JUDUL_TERLALU_PANJANG_KATA} kata). Sederhanakan judul."
    ),
    Rule(
        name="terlalu_umum",
        check=_is_terlalu_umum,
        reason="Judul terlalu umum dan tidak spesifik. Tambahkan metode, objek, dan konteks penelitian."
    ),
    Rule(
        name="tidak_ada_metode",
        check=lambda o, c: not _has_kata_metode(o, c),
        reason="Judul tidak mencantumkan metode atau algoritma yang digunakan."
    ),
    Rule(
        name="tidak_ada_objek",
        check=lambda o, c: not _has_kata_objek(o, c),
        reason="Judul tidak mencantumkan objek atau domain penelitian yang jelas."
    ),
    Rule(
        name="singkatan_berlebihan",
        check=_has_singkatan_tidak_jelas,
        reason="Terlalu banyak singkatan. Tulis kepanjangan singkatan agar judul lebih jelas."
    ),
]


def get_rejection_reason(judul_original: str, judul_clean: str) -> str:
    """
    Evaluasi semua rules dan kembalikan alasan penolakan.
    
    Returns:
        String alasan penolakan, atau pesan diterima jika semua lolos.
    """
    violations = []
    
    for rule in RULES:
        if rule.check(judul_original, judul_clean):
            violations.append(f"• {rule.reason}")
    
    if violations:
        return "Judul ditolak karena:\n" + "\n".join(violations)
    
    return "Judul sudah sesuai. Mengandung metode yang jelas, objek penelitian, dan panjang yang tepat."