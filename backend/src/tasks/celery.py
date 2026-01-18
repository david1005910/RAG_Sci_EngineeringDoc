"""Celery application configuration"""

from celery import Celery
from celery.schedules import crontab

from src.core.config import settings

# Create Celery app
celery_app = Celery(
    "bio_rag",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.REDIS_URL,
    include=[
        "src.tasks.crawler",
        "src.tasks.embedding",
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
)

# Scheduled tasks (Celery Beat)
celery_app.conf.beat_schedule = {
    "daily-paper-crawl": {
        "task": "src.tasks.crawler.daily_paper_crawl",
        "schedule": crontab(hour=2, minute=0),  # Run at 02:00 UTC daily
        "args": (),
    },
}
