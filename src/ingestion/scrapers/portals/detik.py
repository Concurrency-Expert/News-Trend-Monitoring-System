import logging
import re
from typing import List
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from config.settings import settings
from src.ingestion.preprocessor import canonicalize_url, is_source_video
from src.ingestion.scrapers.base import BasePortalStrategy

logger = logging.getLogger(__name__)


class DetikStrategy(BasePortalStrategy):

    def __init__(self):
        super().__init__(source_name="detik")
        self.index_url = settings.NEWS_CONFIGS["detik"]["index_url"]

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

            if "detik.com" not in full_url:
                continue

            # match pola link artikel /d-<id_angka>/
            # cth: https://news.detik.com/berita/d-8759817/kpk-periksa-lagi...
            if not re.search(r"/d-\d+/", full_url):
                continue

            # filter konten video (20.detik.com), galeri foto, atau video tags
            if (
                "20.detik.com" in full_url
                or "/foto/" in full_url
                or "foto.detik.com" in full_url
                or is_source_video(title="", url=full_url)
            ):
                continue

            clean_url = canonicalize_url(full_url)
            if clean_url:
                candidate_urls.add(clean_url)

        urls_list = list(candidate_urls)
        logger.info(f"Successfully fetched {len(urls_list)} articles from {self.source_name}")
        return urls_list
