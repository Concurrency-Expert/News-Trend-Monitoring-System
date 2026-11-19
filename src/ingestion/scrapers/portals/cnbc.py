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

        # Menghapus tag media lain dalam kurung siku
        text = re.sub(
            r"\[\s*(?:video|foto|photo|infografis|embed|sosmed|instagram|twitter|tiktok|youtube)[^\]]*\]",
            "",
            text,
            flags=re.IGNORECASE,
        )

        CNBC_TAIL_MARKERS = [
            "Saksikan video terkait di sini:",
            "Saksikan Video:",
            "Simak Video:",
            "Saksikan video",
            "Simak video",
        ]
        for marker in CNBC_TAIL_MARKERS:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]

        # delete for reporter tag
        while True:
            prev = text.strip()

            cleaned = re.sub(r"\[[^\]]*\]\s*$", "", prev)

            
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