from bs4 import BeautifulSoup
import logging
from typing import List
from urllib.parse import urljoin

from src.ingestion.scrapers.base import BasePortalStrategy
from src.ingestion.preprocessor import canonicalize_url, is_source_video
from config.settings import settings

logger = logging.getLogger(__name__)

class RepublikaStrategy(BasePortalStrategy):

    def __init__(self):
        super().__init__(source_name="republika")
        self.index_url = settings.NEWS_CONFIGS["republika"]["index_url"]

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

            if "republika.co.id" not in full_url:
                continue

            if "/berita/" not in full_url:
                continue

            if is_source_video(title="", url=full_url) or "/inpicture/" in full_url:
                continue

            clean_url = canonicalize_url(full_url)
            if clean_url:
                candidate_urls.add(clean_url)

        urls_list = list(candidate_urls)
        logger.info(f"Successfully fetched {len(urls_list)} articles from {self.source_name}")
        return urls_list
