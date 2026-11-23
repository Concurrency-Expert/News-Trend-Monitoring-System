import logging
from typing import Optional
import pymongo
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from config.settings import settings

logger = logging.getLogger(__name__)


def get_mongo_client() -> pymongo.MongoClient:
    return pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)


def get_database(client: Optional[pymongo.MongoClient] = None) -> Database:
    cli = client or get_mongo_client()
    return cli[settings.MONGO_DB_NAME]


def ensure_indexes(collection: Collection) -> None:
    
    # - url_hash: UNIQUE index (idempotensi)
    # - published_at: DESCENDING index
    # - topic_id: ASCENDING index
    
    try:
        collection.create_index("url_hash", unique=True)
        collection.create_index([("published_at", pymongo.DESCENDING)])
        collection.create_index([("topic_id", pymongo.ASCENDING)])
        logger.info("MongoDB indexes verified successfully.")
    except PyMongoError as e:
        logger.warning(f"Failed to create/verify indexes: {e}")