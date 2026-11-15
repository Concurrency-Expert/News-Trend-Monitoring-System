import logging
from typing import Dict, Type, List
from src.ingestion.scrapers.base import BasePortalStrategy
from src.ingestion.scrapers.portals.kompas import KompasStrategy

logger = logging.getLogger(__name__)

class ScraperFactory:

    NewsStrategy: Dict[str, Type[BasePortalStrategy]] = {
        "kompas": KompasStrategy
    }

    @classmethod # it helps so I don't have to factory = ScraperFactory(); I can automate ScraperFactory.get_strategy("")
    def get_strategy(cls, source_name: str) -> BasePortalStrategy:

        if not source_name or not isinstance(source_name, str):
            raise ValueError("News portal source must be a valid string")

        source_key = source_name.strip().lower()
        strategy_class = cls.NewsStrategy.get(source_key)

        if not strategy_class:
            available_src = ", ".join(cls.NewsStrategy.keys())
            raise ValueError(
                f"News Portal '{source_name}' is not available yet. \n"
                f"News Portals that are on the list: [{available_src}]"
            )
        # it will instansiate the class of newsstrategy
        return strategy_class()

    @classmethod
    def get_supported_sources(cls) -> List[str]:
        return list(cls.NewsStrategy.keys())