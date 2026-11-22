# News Trend Monitoring System — Phase 2: NLP Engine

Kami merancang modul NLP pada Phase 2 ini sebagai mesin klasterisasi berita Indonesia berbasis TF-IDF yang deterministik tanpa ketergantungan LLM. Pendekatan ini kami pilih agar sistem mampu mengelompokkan artikel dari Detik, Republika, CNBC Indonesia, dan Kompas secara cepat, hemat komputasi, dan reproduksibel. Seluruh hasil klasterisasi mengacu langsung pada schema collection `ARTICLES` dan `TOPICS` di MongoDB.

## Alur Kerja Modul NLP

Sebelum membentuk vektor fitur, kami membersihkan teks berita mentah melalui `IndonesianTextPreprocessor` dan daftar `MEDIA_STOPWORDS` pada `src/analytics/text_preprocessor.py`. Modul ini menghapus suffix judul portal, timestamp Kompas, tagline iklan, serta dateline media di awal baris sebelum menyaring stopword PySastrawi dan istilah khas jurnalistik. Proses penyaringan kami jalankan sebelum dan sesudah stemming ber-cache (`_stem_cache`), sambil mencatat pasangan normalized term dan surface token agar penamaan topik nanti tetap natural.

Untuk memproses berita secara bertahap per jadwal batch, kami membangun `IncrementalTfidfVectorizer` di `src/analytics/topic_clusterer.py`. Kami mempertahankan vocabulary append-only hingga batas `max_features` sambil menghitung sublinear TF, smoothed IDF global, dan L2 normalization. Kami juga menyediakan `export_state()` dan `from_state()` untuk menyimpan statistik frekuensi dokumen antar batch, ditambah metode `align_vector()` berbasis zero-padding agar dimensi `centroid_vector` topik lama tetap selaras saat kosakata baru masuk.

Pada tahap klasterisasi di `TopicClusterer`, kami hanya mengambil artikel yang belum memiliki `topic_id` (`topic_id = None`) dan topik dengan `status = "active"`, lalu mengurutkan batch secara deterministik berdasarkan waktu terbit dan kunci artikel. Setiap vektor artikel kami bandingkan terhadap centroid aktif menggunakan cosine similarity dengan ambang batas default `0.60`. Jika cocok, artikel langsung terhubung ke topik tersebut dan centroid diperbarui melalui rata-rata berbobot yang dinormalisasi ulang ke norma L2.

Artikel yang tidak cocok dengan topik aktif kami kelompokkan lebih lanjut memakai `AgglomerativeClustering(metric="cosine", linkage="average")`. Kami mewajibkan ukuran klaster minimal dua artikel agar topik baru terbentuk dengan identitas `topik_YYYYMMDD_NNN`, sementara artikel singleton dibiarkan unmatched agar berpeluang bergabung pada batch berikutnya. Setiap topik baru otomatis kami beri dua kata label dari bobot TF-IDF tertinggi menggunakan surface token paling sering muncul.

## Struktur Berkas dan Konfigurasi

| Berkas | Peran Utama |
| :--- | :--- |
| `config/settings.py` | Menyimpan parameter default konfigurasi `NLP_*` |
| `src/analytics/text_preprocessor.py` | Membersihkan boilerplate portal, stopword media, dan stemming PySastrawi |
| `src/analytics/topic_clusterer.py` | Mengelola TF-IDF incremental, centroid matching, discovery, dan auto-labeling |
| `src/analytics/__init__.py` | Mengekspor API utama modul NLP untuk dipakai oleh layer lain |
| `tests/test_clustering.py` | Berisi 9 unit test dan integration test untuk verifikasi Phase 2 |

Seluruh parameter default dapat disesuaikan melalui class `Settings` di `config/settings.py`:

| Parameter | Nilai Default | Fungsi |
| :--- | :---: | :--- |
| `NLP_SIMILARITY_THRESHOLD` | `0.60` | Ambang batas minimum cosine similarity untuk pencocokan dan klasterisasi |
| `NLP_MIN_CLUSTER_SIZE` | `2` | Jumlah minimum artikel untuk membentuk topik baru |
| `NLP_TOP_KEYWORDS` | `8` | Jumlah kata kunci representatif yang disimpan pada dokumen topik |
| `NLP_LABEL_KEYWORDS` | `2` | Jumlah kata penyusun label topik otomatis |
| `NLP_MAX_FEATURES` | `5000` | Kapasitas maksimum ukuran vocabulary TF-IDF incremental |
| `NLP_TITLE_WEIGHT` | `3` | Faktor pengali bobot token judul relatif terhadap isi berita |
| `NLP_USE_STEMMING` | `True` | Mengaktifkan atau menonaktifkan stemming PySastrawi saat preprocessing |

## Contoh Penggunaan dan Pengujian

Kami merancang antarmuka `TopicClusterer.cluster_articles()` agar tidak terikat langsung pada koneksi database, sehingga mudah diuji secara terisolasi maupun dipanggil dari Airflow DAG. Potongan kode berikut memperlihatkan cara menjalankan satu siklus klasterisasi sekaligus menyimpan state vocabulary untuk batch selanjutnya:

```python
from datetime import datetime, timezone
from src.analytics import IncrementalTfidfVectorizer, TopicClusterer

clusterer = TopicClusterer()
articles = [
    {
        "url_hash": "hash_detik_01",
        "title": "IHSG Ditutup Merah di Sesi I Imbas Saham Big Caps",
        "clean_text": "Indeks Harga Saham Gabungan melemah akibat tekanan saham big caps.",
        "published_at": datetime(2026, 11, 19, 6, 5, tzinfo=timezone.utc),
        "topic_id": None,
    },
    {
        "url_hash": "hash_cnbc_02",
        "title": "Saham Big Caps Tekan IHSG Hingga Merah di Sesi I",
        "clean_text": "Penurunan saham big caps menekan IHSG pada penutupan sesi pertama.",
        "published_at": datetime(2026, 11, 19, 6, 20, tzinfo=timezone.utc),
        "topic_id": None,
    },
]

result = clusterer.cluster_articles(articles, active_topics=[])
print(result.article_topic_assignments)
print(result.new_topics[0]["label"])

# Memuat kembali vectorizer dengan state yang tersimpan untuk batch berikutnya
next_vectorizer = IncrementalTfidfVectorizer.from_state(result.vocabulary_state)
```

Kami menguji seluruh alur preprocessing, alignment dimensi lintas batch, promosi singleton, konsistensi urutan, serta kesesuaian schema dokumen `TOPICS` lewat perintah berikut:

```bash
pytest -v tests/test_clustering.py
```
