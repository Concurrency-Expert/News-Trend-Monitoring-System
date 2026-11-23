from bs4 import BeautifulSoup
import logging
import re
from typing import List
from src.ingestion.scrapers.base import BasePortalStrategy
from src.ingestion.preprocessor import canonicalize_url, is_source_video
from config.settings import settings

logger = logging.getLogger(__name__)

class KompasStrategy(BasePortalStrategy):

    def __init__(self):
        super().__init__(source_name="kompas")
        self.index_url = settings.NEWS_CONFIGS["kompas"]["index_url"]

    def extract_index_urls(self):
        logger.info(f"Retrieveing index from: {self.index_url}")
        html = self.fetch_html(self.index_url)
        if not html:
            return []
        soup = BeautifulSoup(html, "lxml")
        candidate_urls = set()

        for a_tag in soup.find_all("a", href=True):
            raw_url = a_tag["href"]

            if "kompas.com" not in raw_url or "/read/" not in raw_url:
                continue

            if is_source_video(title="", url=raw_url) or "/galeri/" in raw_url:
                continue

            clean_url = canonicalize_url(raw_url)
            if clean_url:
                candidate_urls.add(clean_url)

        urls_list = list(candidate_urls)
        logger.info(f"Successfully fetched {len(urls_list)} articles from {self.source_name}")
        return urls_list

    def clean_portal_noise(self, text):

        if not text:
            return ""
        
        text = re.sub(r"^Baca berita tanpa iklan[^\n]*\n?", "", text.strip(), flags=re.IGNORECASE)

        KOMPAS_TAIL_MARKERS = [
            "Bahas berita ini dengan KARIN",
            "Apresiasi Spesial",
            "Unduh Kompas.com App",
        ]
        for marker in KOMPAS_TAIL_MARKERS:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
        return text.strip()