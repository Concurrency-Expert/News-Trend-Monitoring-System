import hashlib
import re
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from bs4 import BeautifulSoup
from typing import Optional

ID_MONTHS_CONVERSION = {
    "januari": 1, "jan": 1,
    "februari": 2, "feb": 2,
    "maret": 3, "mar": 3,
    "april": 4, "apr": 4,
    "mei": 5,
    "juni": 6, "jun": 6,
    "juli": 7, "jul": 7,
    "agustus": 8, "agu": 8, "agt": 8,
    "september": 9, "sep": 9,
    "oktober": 10, "okt": 10,
    "november": 11, "nov": 11,
    "desember": 12, "des": 12,
}


EN_MONTHS_CONVERSION = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "agu": 8, "aug": 8,
    "september": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

GARBAGE_PARAMS = {
    "utm_source", 
    "utm_medium", 
    "utm_campaign", 
    "utm_term", 
    "utm_content",
    "ref", 
    "source", 
    "platform", 
    "fbclid", 
    "gclid", 
    "_ga"
}

MONTH_LOOKUP = {**ID_MONTHS_CONVERSION, **EN_MONTHS_CONVERSION}

# cleaning url from garbage params
def canonicalize_url(raw_url: str) -> str:
    
    if not raw_url or not isinstance(raw_url, str):
        return ""

    # cleaning whitespace
    raw_url = raw_url.strip()

    # url component parsing
    parsed = urlparse(raw_url)

    # URL-PARSE
    # scheme: https
    # netloc: nasional.tempo.co
    # path: /read/123/jokowi-resmikan-tol
    # query: utm_source=twitter&utm_medium=sosmed 
    # fragment: komentar 

    # scheme normalization

    scheme = parsed.scheme
    if parsed.scheme in ["http", "https"]:
        scheme = "https"
    
    netloc = parsed.netloc.lower()

    # path normalization --> /hehe/123/ menjadi /hehe/123
    path = parsed.path.rstrip("/")

    # query normalization (membersihkan query seperti, "?a=1&b=2 menjadi list pasangan: [('a','1'), ('b','2')]")
    query_tuples = parse_qsl(parsed.query, keep_blank_values=False)

    filtered_queries = []
    for key, value in query_tuples:
        if key.lower() not in GARBAGE_PARAMS:
            filtered_queries.append((key, value))

    # sort query parameter based on key name, contoh: ?kategori=politik&id=1, ?id=1&kategori=politik adalah dua hal yang identik
    filtered_queries.sort(key= lambda x: x[0])

    # wrapped up menjadi string kuery lagi yang sudah ternormalisasi: 'id=1&kategori=politik'
    clean_query = urlencode(filtered_queries)

    clean_url = urlunparse(
        # scheme, netloc, path, params, query, fragment
        (scheme, netloc, path, "", clean_query, "")
    )
    return clean_url

def generate_url_hash(canonical_url: str) -> str:
    # digunakan untuk menghasilkan string SHA-256 dari url yang sudah canonized dan menjadi unique index terhadap url_hash saat disimpan di DB.

    if not canonical_url or not isinstance(canonical_url, str):
        return ""

    # encode to utf-8 karena hashlib.sha256 butuh data dalam bentuk binary
    encoded_url = canonical_url.encode("utf-8")

    # .hexdigest() mengubah hasil hash jadi string 64 karakter (0-9, a-f)
    return hashlib.sha256(encoded_url).hexdigest()

def clean_html_text(data: str) -> str:
    if not data or  not isinstance(data, str):
        return ""

    soup = BeautifulSoup(data, "lxml")

    UNWANTED_TAGS = [
        "script",
        "style",
        "noscript",
        "header",
        "footer",
        "aside",
        "nav",
        "figure"
    ]

    for tag in soup.find_all(UNWANTED_TAGS):
        tag.decompose() # delete tag beserta isinya 

    # fetch text
    text = soup.get_text(separator="\n")

    # change non-breaking space '\xa0' dgn spasi biasa
    text = text.replace("\xa0", " ")

   # change spasi/tab horizontal ganda jadi 1 spasi
    text = re.sub(r"[ \t]+", " ", text)
    
    # sanitize enter berlebih (maksimal 2 enter agar jadi jeda paragraf yang rapi)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    
    return text

def parse_indonesian_date(raw_date: str) -> datetime:
   
    if not raw_date or not isinstance(raw_date, str):
        return datetime.now(timezone.utc)
    
    text = raw_date.strip().lower()
    
    # offset zona waktu lokal (default WIB: UTC+7)
    timezone_offset_hours = 7
    if "wita" in text:
        timezone_offset_hours = 8
    elif "wit" in text:
        timezone_offset_hours = 9
    elif "gmt" in text or "utc" in text:
        timezone_offset_hours = 0
        
    # regex matching
    # tanggal (\d{1,2}) - bulan ([a-z]+) - tahun (\d{4}) - jam (\d{1,2}):menit (\d{2})
    # contohnya: "15 november 2026 14:30" atau "15-nov-2026 14:30:00"
    pattern = r"(\d{1,2})[\s\/-]+([a-z]+)[\s\/-]+(\d{4})[^\d]*(\d{1,2}):(\d{2})(?::(\d{2}))?"
    match = re.search(pattern, text)
    
    if match:
        day = int(match.group(1))
        month_str = match.group(2)
        year = int(match.group(3))
        hour = int(match.group(4))
        minute = int(match.group(5))
        second = int(match.group(6)) if match.group(6) else 0
        
        # fetch angka bulan dari MONTH_LOOKUP 
        month = MONTH_LOOKUP.get(month_str)
        if month:
            try:
                # bentuk objek datetime lokal based on timezone aslinya
                local_tz = timezone(timedelta(hours=timezone_offset_hours))
                local_dt = datetime(year, month, day, hour, minute, second, tzinfo=local_tz)
                
                # konversi ke utc time
                return local_dt.astimezone(timezone.utc)
            except ValueError:
                pass

    # jika regex tidak matched, pake parser dari library dateutil
    try:
        from dateutil import parser
        dt = parser.parse(raw_date)
        if dt.tzinfo is None:
            # jika tdk ada info zona waktu, pake default wib
            dt = dt.replace(tzinfo=timezone(timedelta(hours=7)))
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def sanitize_common_noise(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"Baca juga:[^\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"BACA JUGA:[^\n]*", "", text)
    text = re.sub(r"Simak juga:[^\n]*", "", text, flags=re.IGNORECASE)
    
    lines = []
    for line in text.split("\n"):
        if line.strip():
            lines.append(line.strip())
            
    return "\n\n".join(lines)

def is_source_video(title: str, url: str) -> bool:

    title_lower = (title or "").lower()
    url_lower = (url or "").lower()

    # title matching
    video_keywords = [
        "video:", 
        "[video]", 
        "(video)", 
        "video -", 
        "live streaming"
    ]

    for keyword in video_keywords:
        if keyword in title_lower:
            return True

    # url matching
    url_patterns = [
        "/video/", 
        "/video-", 
        "video.", 
        "/tv/", 
        "/watch/"
    ]
    for pattern in url_patterns:
        if pattern in url_lower:
            return True

    
    return False