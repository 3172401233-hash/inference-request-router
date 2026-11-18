"""推理请求路由器。

基于副本队列长度、显存页余量、KV 前缀命中率、版本约束与降级策略，
为每条推理请求选择一个目标副本，并给出选择原因与预计排队风险。
"""

from router.models import (
    QueueRisk,
    Replica,
    ReplicaStatus,
    Request,
    RiskLevel,
    RouteReason,
    RouterConfig,
    RoutingDecision,
)
from router.router import (
    IncompatibleModelError,
    InferenceRouter,
    InsufficientCapacityError,
    NoReplicaAvailableError,
)

__all__ = [
    "QueueRisk",
    "Replica",
    "ReplicaStatus",
    "Request",
    "RiskLevel",
    "RouteReason",
    "RouterConfig",
    "RoutingDecision",
    "IncompatibleModelError",
    "InferenceRouter",
    "InsufficientCapacityError",
    "NoReplicaAvailableError",
]
