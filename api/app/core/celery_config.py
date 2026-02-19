from celery import Celery
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "openmemoryx",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.services.memory_queue"]
)

celery_app.conf.update(
    # 任务序列化
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # 时区
    timezone="UTC",
    enable_utc=True,
    
    # 并发设置 - 限制同时处理2个（匹配Ollama能力）
    worker_concurrency=2,  # 2个并发worker
    worker_prefetch_multiplier=1,  # 每个worker只预取1个任务
    
    # 任务超时
    task_time_limit=300,  # 5分钟超时
    task_soft_time_limit=240,  # 4分钟软超时
    
    # 重试机制
    task_max_retries=3,
    task_default_retry_delay=10,  # 10秒后重试
    
    # 结果存储
    result_expires=3600,  # 结果保留1小时
    
    # 队列配置
    task_default_queue="memory",
    task_routes={
        "app.services.memory_queue.add_memory_task": {"queue": "memory"},
        "app.services.memory_queue.batch_add_memory_task": {"queue": "memory"},
        "app.services.memory_queue.update_memory_task": {"queue": "memory"},
        "app.services.memory_queue.delete_memory_task": {"queue": "memory"},
    },
)

# 定义队列
celery_app.conf.task_queues = {
    "memory": {
        "exchange": "memory",
        "routing_key": "memory",
    },
}
