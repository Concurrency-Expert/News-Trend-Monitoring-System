"""Indonesian journalistic text preprocessing for TF-IDF topic clustering."""

from __future__ import annotations

import re
from typing import Iterable, Optional

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

MEDIA_STOPWORDS: frozenset[str] = frozenset(
    {
        # Portal identifiers & domains
        "detik",
        "detikcom",
        "detiknews",
        "detikfinance",
        "detikhot",
        "detikinet",
        "detikoto",
        "detiksport",
        "republika",
        "kompas",
        "kompascom",
        "cnbc",
        "cnbcindonesia",
        "tempo",
        "tempoco",
        "online",
        "com",
        "co",
        "id",
        "www",
        "http",
        "https",
        # Timezones, days, months, temporal fillers
        "wib",
        "wita",
        "wit",
        "gmt",
        "utc",
        "senin",
        "selasa",
        "rabu",
        "kamis",
        "jumat",
        "sabtu",
        "minggu",
        "januari",
        "februari",
        "maret",
        "april",
        "mei",
        "juni",
        "juli",
        "agustus",
        "september",
        "oktober",
        "november",
        "desember",
        "hari",
        "tanggal",
        "bulan",
        "tahun",
        "pekan",
        "kemarin",
        "besok",
        "pukul",
        "siang",
        "malam",
        "pagi",
        "sore",
        # Common dateline / web boilerplate
        "jakarta",
        "baca",
        "simak",
        "selengkapnya",
        "halaman",
        "berita",
        "artikel",
        "foto",
        "video",
        "redaksi",
        "editor",
        "penulis",
        "reporter",
        "sumber",
        "dikutip",
        "dilansir",
        # Journalistic attribution & filler tokens
        "kata",
        "katanya",
        "ujar",
        "ujarnya",
        "tutur",
        "tuturnya",
        "ungkap",
        "ungkapnya",
        "jelas",
        "jelasnya",
        "sebut",
        "sebutnya",
        "mengatakan",
        "menyatakan",
        "menjelaskan",
        "mengungkapkan",
        "menyebut",
        "menyebutkan",
        "menambahkan",
        "menegaskan",
        "menuturkan",
        "menilai",
        "menurut",
        "menurutnya",
        "disampaikan",
        "dikatakan",
        "dijelaskan",
        "tersebut",
        "yakni",
        "yaitu",
        "terkait",
        "mengenai",
        "usai",
        "soal",
        "bakal",
        "pihak",
        "orang",
        "hal",
    }
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+")
_TITLE_SUFFIX_RE = re.compile(
    r"\s*[|\-–—]\s*(?:republika\s+online|kompas\.com|detik(?:news|finance|hot|inet|oto|sport|edu|health|food|com)?|cnbc\s+indonesia|tempo\.co)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TAGLINE_RE = re.compile(
    r"^\s*tumbuh bersama kekuatan mimpi perempuan indonesia\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_KOMPAS_TIMESTAMP_RE = re.compile(
    r"^\s*kompas\.com,\s*\d{1,2}\s+[a-z]+\s+\d{4}(?:,\s*\d{1,2}:\d{2}\s*(?:wib|wita|wit))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_REPUBLIKA_DATELINE_RE = re.compile(
    r"^\s*republika\.co\.id(?:\s*,\s*[a-z]{2,20}(?:\s+[a-z]{2,20})?)?\s*,?\s*[-–—]+\s*",
    re.IGNORECASE | re.MULTILINE,
)
_PORTAL_DATELINE_RE = re.compile(
    r"^\s*(?:[a-z]{2,20}(?:\s+[a-z]{2,20})?,\s*)?(?:kompas\.com|cnbc\s+indonesia|tempo\.co|detikcom)\s*[-–—:]\s*",
    re.IGNORECASE | re.MULTILINE,
)
_CITY_DATELINE_RE = re.compile(
    r"^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+[-–—]\s+",
    re.MULTILINE,
)
_TOKEN_RE = re.compile(r"[a-z]+")


class IndonesianTextPreprocessor:
    """Clean, filter stopwords, and stem Indonesian news articles for TF-IDF."""

    def __init__(
        self,
        use_stemming: bool = True,
        extra_stopwords: Optional[Iterable[str]] = None,
        min_token_length: int = 2,
    ) -> None:
        if min_token_length < 1:
            raise ValueError("min_token_length must be >= 1")

        self.use_stemming = use_stemming
        self.min_token_length = min_token_length

        sastrawi_stopwords = {
            w.strip().lower() for w in StopWordRemoverFactory().get_stop_words()
        }
        custom = {w.strip().lower() for w in (extra_stopwords or []) if w.strip()}
        self.stopwords: frozenset[str] = frozenset(
            sastrawi_stopwords | MEDIA_STOPWORDS | custom
        )

        self._stemmer = StemmerFactory().create_stemmer() if use_stemming else None
        self._stem_cache: dict[str, str] = {}

    def strip_portal_boilerplate(self, text: Optional[str]) -> str:
        """Remove portal headers, timestamps, datelines, URLs, and emails."""
        if not text or not isinstance(text, str):
            return ""

        cleaned = _URL_RE.sub(" ", text)
        cleaned = _EMAIL_RE.sub(" ", cleaned)
        cleaned = _TITLE_SUFFIX_RE.sub("", cleaned)
        cleaned = _TAGLINE_RE.sub("", cleaned)
        cleaned = _KOMPAS_TIMESTAMP_RE.sub("", cleaned)
        cleaned = _REPUBLIKA_DATELINE_RE.sub("", cleaned)
        cleaned = _PORTAL_DATELINE_RE.sub("", cleaned)
        cleaned = _CITY_DATELINE_RE.sub("", cleaned)
        return cleaned.strip()

    def _stem_word(self, token: str) -> str:
        """Stem a lowercase token using PySastrawi with memoization."""
        if not self._stemmer:
            return token
        cached = self._stem_cache.get(token)
        if cached is not None:
            return cached
        stemmed = self._stemmer.stem(token) or token
        self._stem_cache[token] = stemmed
        return stemmed

    def tokenize_with_surfaces(self, text: Optional[str]) -> list[tuple[str, str]]:
        """Return `(normalized_term, surface_token)` pairs after stopword & stem filtering."""
        cleaned = self.strip_portal_boilerplate(text)
        if not cleaned:
            return []

        raw_tokens = _TOKEN_RE.findall(cleaned.lower())
        pairs: list[tuple[str, str]] = []
        for surface in raw_tokens:
            if len(surface) < self.min_token_length or surface in self.stopwords:
                continue
            normalized = self._stem_word(surface) if self.use_stemming else surface
            if len(normalized) < self.min_token_length or normalized in self.stopwords:
                continue
            pairs.append((normalized, surface))
        return pairs

    def preprocess(self, text: Optional[str]) -> str:
        """Normalize raw text into space-separated terms ready for TF-IDF."""
        return " ".join(term for term, _ in self.tokenize_with_surfaces(text))

    def preprocess_article(
        self,
        title: Optional[str],
        clean_text: Optional[str],
        title_weight: int = 3,
    ) -> tuple[str, list[tuple[str, str]]]:
        """Preprocess an article combining weighted title tokens and body text."""
        if title_weight < 0:
            raise ValueError("title_weight must be >= 0")

        title_pairs = self.tokenize_with_surfaces(title) if title_weight > 0 else []
        body_pairs = self.tokenize_with_surfaces(clean_text)

        weighted_pairs: list[tuple[str, str]] = []
        if title_weight > 0 and title_pairs:
            weighted_pairs.extend(title_pairs * title_weight)
        weighted_pairs.extend(body_pairs)

        processed_text = " ".join(term for term, _ in weighted_pairs)
        return processed_text, weighted_pairs
