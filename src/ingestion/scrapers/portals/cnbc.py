import logging
import re
from typing import List
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from config.settings import settings
from src.ingestion.preprocessor import canonicalize_url, is_source_video
from src.ingestion.scrapers.base import BasePortalStrategy

logger = logging.getLogger(__name__)


class CNBCStrategy(BasePortalStrategy):

    def __init__(self):
        super().__init__(source_name="cnbc")
        self.index_url = settings.NEWS_CONFIGS["cnbc"]["index_url"]

    def extract_index_urls(self) -> List[str]:
        logger.info(f"Retrieveing index from: {self.index_url}")
        html = self.fetch_html(self.index_url)
        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")
        candidate_urls = set()

        for a_tag in soup.find_all("a", href=True):
            raw_url = a_tag["href"].strip()
            full_url = urljoin(self.index_url, raw_url)
            
            if "cnbcindonesia.com" not in full_url:
                continue

            # url CNBC ada format timestamp unik: /\d{8,14}-\d+-\d+/
            # cth: https://www.cnbcindonesia.com/news/20261119101856-4-786531/polisi-sita..
            if not re.search(r"/\d{8,14}-\d+-\d+/", full_url):
                continue

            if (
                is_source_video(title="", url=full_url)
                or "/video-" in full_url
                or "/foto/" in full_url
                or "/opini/" in full_url
            ):
                continue

            clean_url = canonicalize_url(full_url)
            if clean_url:
                candidate_urls.add(clean_url)

        urls_list = list(candidate_urls)
        logger.info(f"Successfully fetched {len(urls_list)} articles from {self.source_name}")
        return urls_list

    def clean_portal_noise(self, text: str) -> str:
        if not text:
            return ""
        
        # delete kurung siku yang memuat indikasi video, link, foto, atau media
        text = re.sub(
            r"\[[^\]]*(?:video|link|http|www|foto|gambar|media|gambas)[^\]]*\]",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # delete link URL mentah yang tertinggal dari embed
        text = re.sub(r"https?://\S+", "", text)
        
        CNBC_TAIL_MARKERS = [
            "saksikan video",
            "simak video",
            "tonton video",
            "video terkait",
        ]
        text_lower = text.lower()
        min_idx = -1
        for marker in CNBC_TAIL_MARKERS:
            idx = text_lower.find(marker)
            if idx != -1:
                if min_idx == -1 or idx < min_idx:
                    min_idx = idx
        if min_idx != -1:
            text = text[:min_idx]
            
        while True:
            prev = text.strip()
            cleaned = re.sub(r"\[[^\]]*\]\s*$", "", prev)
            cleaned = re.sub(
                r"(?:\n|\A)\s*video[^\n]*$", "", cleaned, flags=re.IGNORECASE
            )
            # delete inisial reporter
            
            cleaned = re.sub(
                r"\(\s*[a-zA-Z\s]{2,20}(?:\s*/\s*[a-zA-Z\s]{2,20})*\s*\)[;\.:\-]*\s*$",
                "",
                cleaned.strip(),
                flags=re.IGNORECASE,
            )
            if cleaned == prev:
                break
            text = cleaned

        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()