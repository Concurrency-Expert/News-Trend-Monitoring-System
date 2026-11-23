from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, Any

class Settings(BaseSettings):

    # MONGO DB CONFIGURATION
    MONGO_URI : str = Field(default="mongodb://localhost:27018")
    MONGO_DB_NAME : str = Field(default="news_monitoring_db")

    # HTTP CLIENT CONFIGURATION
    # use for agar tidak terdetect sebagai bot di firewall
    DEFAULT_HEADERS: Dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            "AppleWebKit/537.36 (KHTML, like Gecko)"
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    REQUEST_TIMEOUT: int = 15
    MAX_RETRIES: int = 3

    # ENDPOINT PORTAL BERITA    
    NEWS_CONFIGS: Dict[str, Dict[str, Any]] = {
        "detik": {
            "name": "detik.com",
            "rss_url": None,
            "index_url": "https://news.detik.com/indeks",
        },
        "republika": {
            "name": "Republika",
            "rss_url": "https://www.republika.co.id/rss/nasional",
            "index_url": "https://www.republika.co.id/index.aspx",
        },
        "cnbc": {
            "name": "CNBC Indonesia",
            "rss_url": "https://www.cnbcindonesia.com/news/rss",
            "index_url": "https://www.cnbcindonesia.com/indeks/",
        },
        "kompas": {
            "name": "Kompas.com",
            "rss_url": None,
            "index_url": "https://indeks.kompas.com/",
        }
    }

    class Config:
        env_file = ".env"
        extra = "ignore"
        
settings = Settings()