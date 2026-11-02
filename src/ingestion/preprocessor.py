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

    # scheme normalization

    scheme = parsed.scheme
    if parsed.scheme in ["http", "https"]:
        scheme = "https"
    
    netloc = parsed.netloc.lower()

    # path normalization --> /hehe/123/ menjadi /hehe/123
    path = parsed.path.rstrip("/")

    #