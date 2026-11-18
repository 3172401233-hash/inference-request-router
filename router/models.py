"""路由器使用的数据模型与配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ReplicaStatus(str, Enum):
    """副本生命周期状态。"""

    ONLINE = "online"
    DRAINING = "draining"
    OFFLINE = "offline"


class RiskLevel(str, Enum):
    """预计排队风险等级。"""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RouteReason(str, Enum):
    """路由选择的主要原因。"""

    ROUND_ROBIN = "round_robin"
    LOWEST_LOAD = "lowest_load"
    KV_AFFINITY = "kv_affinity"
    IDLE_STARVATION = "idle_starvation"
    DOWNGRADE = "downgrade"


@dataclass(frozen=True)
class Request:
    """一条推理请求。

    ``deadline_ms`` 为绝对时间戳（毫秒）；``kv_prefix_hits`` 以
    ``副本 id -> 前缀命中率（0.0~1.0）`` 描述各副本的 KV 亲和度。
    """

    request_id: str
    model: str
    max_tokens: int
    deadline_ms: int
    prompt_tokens: int = 0
    kv_prefix_hits: Mapping[str, float] = field(default_factory=dict)
    downgrade_allowed: bool = False
    downgrade_models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须为正整数")
        if self.deadline_ms < 0:
            raise ValueError("deadline_ms 不能为负")
        for replica_id, hit in self.kv_prefix_hits.items():
            if not 0.0 <= hit <= 1.0:
                raise ValueError(f"副本 {replica_id} 的 KV 命中率必须落在 [0, 1]")


@dataclass
class Replica:
    """一个模型副本的实时状态。"""

    replica_id: str
    model: str
    status: ReplicaStatus = ReplicaStatus.ONLINE
    queue_depth: int = 0
    avg_prefill_ms: float = 20.0
    avg_decode_per_token_ms: float = 10.0
    total_pages: int = 0
    free_pages: int = 0
    last_dispatched_ms: int | None = None

    def __post_init__(self) -> None:
        if self.queue_depth < 0:
            raise ValueError("queue_depth 不能为负")
        if self.total_pages < 0 or self.free_pages < 0:
            raise ValueError("显存页数不能为负")
        if self.free_pages > self.total_pages:
            raise ValueError("free_pages 不能大于 total_pages")
        if self.avg_prefill_ms < 0 or self.avg_decode_per_token_ms < 0:
            raise ValueError("平均时延不能为负")

    @property
    def is_available(self) -> bool:
        """只有 ONLINE 副本承接新流量。"""
        return self.status is ReplicaStatus.ONLINE

    @property
    def load_ratio(self) -> float:
        """队列相对容量的占用比例，用于公平性比较。"""
        if self.total_pages <= 0:
            return float(self.queue_depth)
        used = self.total_pages - self.free_pages
        return used / self.total_pages


@dataclass(frozen=True)
class QueueRisk:
    """目标副本上的预计排队风险。"""

    level: RiskLevel
    estimated_wait_ms: float
    slack_ms: float
    reason: str

    @property
    def deadline_met(self) -> bool:
        return self.slack_ms > 0


@dataclass(frozen=True)
class RoutingDecision:
    """路由结果。"""

    request_id: str
    replica_id: str
    model: str
    downgraded: bool
    reason: RouteReason
    reason_detail: str
    score: float
    score_breakdown: Mapping[str, float]
    risk: QueueRisk
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class RouterConfig:
    """路由器权重与阈值配置。

    总分越高越优先。亲和项（``weight_kv_affinity``）提升 KV 命中率高的
    副本，而饥饿保护项（``weight_starvation``）保证长期空闲副本不会因为
    缺少亲和分而永远落选，二者共同决定亲和性与公平性的平衡。
    """

    page_size_tokens: int = 16
    prompt_page_multiplier: float = 1.0
    weight_queue_risk: float = 1.0
    weight_load: float = 0.5
    weight_memory_pressure: float = 0.3
    weight_kv_affinity: float = 0.8
    weight_starvation: float = 1.2
    idle_empty_bonus: float = 0.15
    starvation_threshold_ms: int = 2_000
    downgrade_risk_threshold: RiskLevel = RiskLevel.HIGH
    high_risk_ratio: float = 0.6
    critical_risk_ratio: float = 0.9

    def __post_init__(self) -> None:
        if self.page_size_tokens <= 0:
            raise ValueError("page_size_tokens 必须为正整数")
        if self.starvation_threshold_ms <= 0:
            raise ValueError("starvation_threshold_ms 必须为正数")
        if not 0.0 <= self.high_risk_ratio <= self.critical_risk_ratio:
            raise ValueError("风险阈值需满足 0 <= high <= critical")
