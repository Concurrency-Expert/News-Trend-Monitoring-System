from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class ArticleDocument(BaseModel):
    url_hash: str
    title: str
    clean_text: str
    url: str
    source: str
    published_at: datetime
    topic_id: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )