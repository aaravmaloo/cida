from dataclasses import dataclass

from redis.asyncio import Redis


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_seconds: int


async def enforce_sliding_window(
    redis: Redis,
    *,
    key: str,
    limit: int,
    window_seconds: int,
) -> RateLimitResult:
    now_ms = int(__import__("time").time() * 1000)
    window_ms = window_seconds * 1000
    cutoff = now_ms - window_ms

    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, 0, cutoff)
    pipe.zcard(key)
    pipe.zadd(key, {str(now_ms): now_ms})
    pipe.expire(key, window_seconds)
    _, count, _, _ = await pipe.execute()

    used = int(count) + 1
    allowed = used <= limit
    remaining = max(0, limit - used)
    return RateLimitResult(allowed=allowed, remaining=remaining, reset_seconds=window_seconds)

