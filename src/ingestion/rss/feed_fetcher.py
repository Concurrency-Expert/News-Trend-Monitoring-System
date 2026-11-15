import feedparser
import logging
from typing import List, Dict, Any

from config.settings import settings
from src.ingestion.preprocessor import (
    canonicalize_url,
    generate_url_hash,
    parse_indonesian_date,
    is_source_video
)

logger = logging.getLogger(__name__)

class RSSFeedFetcher:

    def __init__(self):
        self.portal_configs = settings.NEWS_CONFIGS

    def fetch_feeds_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        
        config = self.portal_configs.get(source_name)
        if not config or not config.get("rss_url"):
            logger.info(f"Media '{source_name}' doesn't have any RSS URL or the url is inactive")
            return []

        rss_url = config["rss_url"]
        logger.info(f"Fetching RSS Feed from: {rss_url}")

        feed = feedparser.parse(rss_url, request_headers=settings.DEFAULT_HEADERS)

        # check whether theres an error while parsing
        if feed.bozo:
            logger.warning(f"Warning feedparser {source_name}: {feed.bozo_exception}")

        articles = []
        for entry in feed.entries:

            raw_title = getattr(entry, "title", "").strip()
            raw_url = getattr(entry, "link", "").strip()
            raw_pub_date = getattr(entry, "published", "")

            if is_source_video(raw_title, raw_url):
                continue

            clean_url = canonicalize_url(raw_url)
            if not clean_url:
                continue
            
            url_hash = generate_url_hash(clean_url)

            published_at = parse_indonesian_date(raw_pub_date)

            articles.append({
                "url_hash": url_hash,
                "url": clean_url,
                "title": raw_title,
                "source": source_name,
                "published_at": published_at,
            })
        logger.info(f"Successfully fetching {len(articles)} articles from RSS: {source_name}")
        return articles

    def fetch_all_active_feeds(self) -> List[Dict[str, Any]]:
        all_articles = []
        for source, config in self.portal_configs.items():
            if config.get("rss_url"):
                portal_articles = self.fetch_feeds_by_source(source_name=source)
                all_articles.extend(portal_articles)
        return all_articles