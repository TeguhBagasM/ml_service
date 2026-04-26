import re
import nltk
from nltk.corpus import stopwords

# Pastikan stopwords sudah didownload
nltk.download('stopwords', quiet=True)

# Stopwords bahasa Indonesia (gabungan NLTK + custom)
STOPWORDS_ID = set(stopwords.words('indonesian'))

# Tambahan stopwords custom domain skripsi
CUSTOM_STOPWORDS = {
    'menggunakan', 'berbasis', 'dengan', 'pada', 'untuk',
    'dalam', 'dan', 'atau', 'di', 'ke', 'dari', 'yang',
    'adalah', 'ini', 'itu', 'sebagai', 'oleh', 'antara',
    'studi', 'kasus', 'study', 'case', 'the', 'of', 'a',
    'rancang', 'bangun', 'perancangan', 'pembangunan'
}

STOPWORDS_ID.update(CUSTOM_STOPWORDS)


def preprocess_text(text: str) -> str:
    """
    Pipeline preprocessing teks judul skripsi.
    
    Args:
        text: String judul skripsi
    
    Returns:
        String teks yang sudah bersih
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # 3. Hapus tanda baca dan karakter khusus
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 4. Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Tokenisasi sederhana
    tokens = text.split()
    
    # 6. Hapus stopwords
    tokens = [word for word in tokens if word not in STOPWORDS_ID]
    
    # 7. Hapus token terlalu pendek (< 2 karakter)
    tokens = [word for word in tokens if len(word) > 2]
    
    return ' '.join(tokens)


def preprocess_batch(texts: list) -> list:
    """Preprocessing untuk list of texts (batch)."""
    return [preprocess_text(t) for t in texts]