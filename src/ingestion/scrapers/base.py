from abc import ABC, abstractmethod
import requests
import trafilatura
import logging
from typing import List, Dict, Any, Optional
from config.settings import settings
from src.ingestion.preprocessor import clean_html_text
logger = logging.getLogger(__name__)

class BasePortalStrategy(ABC):

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.session = requests.Session()
        self.session.headers.update(settings.DEFAULT_HEADERS)
        self.timeout = settings.REQUEST_TIMEOUT

    def fetch_html(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"Failed to retrieve {url} - Status Code: {response.status_code}")
                return None
        except requests.RequestException as e:
            logger.error(f"Network error while retrieving {url}: {e}")
            return None

    def scrape_article_detail(self, url: str) -> Optional[Dict[str, Any]]:
        
        html = self.fetch_html(url=url)
        if not html:
            return None

        extracted_text = trafilatura.extract(
            html,
            include_comments=False, # exclude comment sections
            include_tables=False, # exclude ads
            no_fallback=False
        )

        metadata = trafilatura.extract_metadata(html)

        if not extracted_text or len(extracted_text.strip()) < 150:
            logger.info(f"Article is skipped by the system because the length of text is too short / there's no text: {url}")
            return None
        
        clean_text = clean_html_text(extracted_text)
        title = metadata.title if (metadata and metadata.title) else None

        return {
            "title": title,
            "clean_text": clean_text
        }

    @abstractmethod
    def extract_index_urls(self) -> List[str]:
        # method to retrieve new articles from the newspage
        pass