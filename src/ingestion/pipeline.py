from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, Optional
import pymongo
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, PyMongoError
from src.ingestion.preprocessor import generate_url_hash
from src.ingestion.rss.feed_fetcher import RSSFeedFetcher
from src.database.mongo_client import ensure_indexes, get_database
from src.api.schemas import ArticleDocument
from src.ingestion.scrapers.factory import ScraperFactory
logger = logging.getLogger(__name__)


class NewsIngestionPipeline:

    def __init__(self, mongo_client: Optional[pymongo.MongoClient] = None):
        self.rss_fetcher = RSSFeedFetcher()
        self.supported_sources = ScraperFactory.get_supported_sources()

        self.db = get_database(client=mongo_client)
        self.collection: Collection = self.db["articles"]
        ensure_indexes(self.collection)

    

    def run_portal(self, source: str) -> Dict[str, Any]:

        # ingestion: discover -> in-memory deduplication -> prefetch filter -> scraping -> save
        logger.info(f"[{source}] Starting ingestion pipeline...")
        candidate_map: Dict[str, Dict[str, Any]] = {}

        # discovery: RSS fetching
        try:
            rss_articles = self.rss_fetcher.fetch_feeds_by_source(source)
            for item in rss_articles:
                candidate_map[item["url_hash"]] = item
            logger.info(f"[{source}] Found {len(rss_articles)} articles in RSS")
        except Exception as e:
            logger.warning(f"[{source}] RSS fetch error: {e}")

         # discovery: Scraping index news
        
        try:
            strategy = ScraperFactory.get_strategy(source)
            index_urls = strategy.extract_index_urls()
            for url in index_urls:
                url_hash = generate_url_hash(url)
                # in-memory deduplication: Check whether theres no candidate url on candidate_map
                if url_hash not in candidate_map:
                    candidate_map[url_hash] = {
                        "url_hash": url_hash,
                        "url": url,
                        "title": None, # judul nanti di-scrape
                        "source": source,
                        "published_at": None,
                    }
            logger.info(
                f"[{source}] Found {len(index_urls)} articles in web index"
            )
        except Exception as e:
            logger.error(f"[{source}] Web index fetch error: {e}")

        
        all_hashes = list(candidate_map.keys())
        if not all_hashes:
            logger.info(f"[{source}] No articles discovered.")
            return {"source": source, "discovered": 0, "saved": 0}
        

        # prefetch filter: cek url-url hash yang ada di all_hashes apakah ada di mongodb atau tidak
        try:
            cursor = self.collection.find(
                {"url_hash": {"$in": all_hashes}}, 
                {"url_hash": 1, "_id": 0}
            )
            existing_url_hashes = {cur["url_hash"] for cur in cursor}

        except PyMongoError as e:
            logger.error(f"[{source}] Pre-fetch filter query error: {e}")
            existing_url_hashes = set()

        # filter url yang benar-benar baru

        new_url_candidates = []
        for key, value in candidate_map.items():
            if key not in existing_url_hashes:
                new_url_candidates.append(value)

        logger.info(
            f"[{source}] Pre-fetch filter: {len(existing_url_hashes)} already exist in DB. "
            f"{len(new_url_candidates)} new articles to scrape."
        )

        # if the url candidates already in db, end program
        if not new_url_candidates:
            return {
                "source": source,
                "discovered": len(candidate_map),
                "saved": 0,
            }

        # scraping: scrape detail news based on the new url candidates

        strategy = ScraperFactory.get_strategy(source)
        saved_count = 0
        for item in new_url_candidates:
            url = item["url"]
            try:
                detail = strategy.scrape_article_detail(url)
                if not detail or not detail.get("clean_text"):
                    logger.warning(
                        f"[{source}] Empty text or failed scraping: {url}"
                    )
                    continue
               
                final_title = (
                    detail.get("title") or item.get("title") or "Tanpa Judul"
                )
                final_published_at = item.get(
                    "published_at"
                ) or datetime.now(timezone.utc)

                # validasi format dokumen sesuai skema MongoDB
                doc = ArticleDocument(
                    url_hash=item["url_hash"],
                    title=final_title.strip(),
                    clean_text=detail["clean_text"],
                    url=url,
                    source=source,
                    published_at=final_published_at,
                    topic_id=None,
                )

                self.collection.insert_one(doc.model_dump())
                saved_count += 1
                time.sleep(0.2)  
            except DuplicateKeyError:
                pass
            except Exception as e:
                logger.error(f"[{source}] Error saving article {url}: {e}")
        logger.info(
            f"[{source}] Finished. Successfully saved {saved_count} new articles to DB."
        )
        return {
            "source": source,
            "discovered": len(candidate_map),
            "new_to_scrape": len(new_url_candidates),
            "saved": saved_count,
        }

    # pipeline runner

    def run(self) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("Starting news ingestion pipeline for all portals...")
        total_saved = 0
        total_discovered = 0
        portal_metrics = {}
 
        for source in self.supported_sources:
            metrics = self.run_portal(source)
            portal_metrics[source] = metrics
            total_saved += metrics.get("saved", 0)
            total_discovered += metrics.get("discovered", 0)

        elapsed = time.time() - start_time

        logger.info(
            f"Pipeline finished in {elapsed:.2f}s. "
            f"Discovered: {total_discovered}, Saved: {total_saved} new articles."
        )

        return {
            "total_discovered": total_discovered,
            "total_saved": total_saved,
            "elapsed_seconds": round(elapsed, 2),
            "details": portal_metrics,
        }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    pipeline = NewsIngestionPipeline()
    pipeline.run()