"""推理请求路由器核心实现。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from router.models import (
    QueueRisk,
    Replica,
    Request,
    RiskLevel,
    RouteReason,
    RouterConfig,
    RoutingDecision,
)


class RouterError(Exception):
    """路由器基础异常。"""


class NoReplicaAvailableError(RouterError):
    """没有任何在线副本可承接请求。"""


class IncompatibleModelError(RouterError):
    """在线副本均不支持请求的模型版本。"""


class InsufficientCapacityError(RouterError):
    """所有候选副本的显存页都不足以容纳请求。"""


@dataclass(frozen=True)
class _Evaluation:
    replica: Replica
    score: float
    breakdown: dict[str, float]
    risk: QueueRisk


def _default_clock_ms() -> int:
    return int(time.monotonic() * 1000)


class InferenceRouter:
    """根据实时副本状态选择目标副本。

    评分越高越优先。评分由 deadline 紧迫度、负载、显存压力、KV 亲和
    与空闲饥饿保护组成；同分时使用轮转游标做 round-robin，保证均匀分发。
    """

    def __init__(
        self,
        replicas: Iterable[Replica],
        config: RouterConfig | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        replica_list = list(replicas)
        ids = [r.replica_id for r in replica_list]
        if len(ids) != len(set(ids)):
            raise ValueError("副本 id 必须唯一")
        self._replicas: dict[str, Replica] = {r.replica_id: r for r in replica_list}
        self._config = config or RouterConfig()
        self._clock_ms = clock_ms or _default_clock_ms
        self._cursor = 0
        boot_ms = self._clock_ms()
        for replica in self._replicas.values():
            if replica.last_dispatched_ms is None:
                replica.last_dispatched_ms = boot_ms

    @property
    def config(self) -> RouterConfig:
        return self._config

    def update_replica(self, replica: Replica) -> None:
        """替换某个副本的最新状态（用于副本下线/上线/指标刷新）。"""
        self._replicas[replica.replica_id] = replica

    def required_pages(self, request: Request) -> int:
        """请求所需显存页数：提示词与最大生成都按页对齐。"""
        tokens = request.prompt_tokens + request.max_tokens
        tokens = int(math.ceil(tokens * self._config.prompt_page_multiplier))
        return max(1, math.ceil(tokens / self._config.page_size_tokens))

    def route(self, request: Request) -> RoutingDecision:
        """为请求计算目标副本、选择原因与预计排队风险。"""
        now_ms = self._clock_ms()
        needed_pages = self.required_pages(request)

        primary_online = [
            r for r in self._replicas.values()
            if r.is_available and r.model == request.model
        ]
        if not primary_online:
            if any(r.model == request.model for r in self._replicas.values()):
                raise NoReplicaAvailableError(
                    f"模型 {request.model} 没有 ONLINE 副本（可能已下线或正在 drain）"
                )
            raise IncompatibleModelError(
                f"在线副本均不支持模型版本 {request.model}"
            )

        primary_fit = [r for r in primary_online if r.free_pages >= needed_pages]
        downgraded = False
        candidates: Sequence[Replica] = primary_fit

        if not primary_fit or self._should_downgrade(request, primary_fit, now_ms):
            fallback = self._downgrade_candidates(request, needed_pages)
            if fallback:
                candidates = fallback
                downgraded = True
            elif not primary_fit:
                raise InsufficientCapacityError(
                    f"请求 {request.request_id} 需要 {needed_pages} 页，"
                    f"模型 {request.model} 的在线副本显存均不足"
                )

        evaluations = [
            self._evaluate(request, replica, needed_pages, now_ms)
            for replica in candidates
        ]
        winner, tie = self._pick_winner(evaluations)
        reason = self._reason(winner, tie, downgraded)

        winner.replica.last_dispatched_ms = now_ms
        self._cursor += 1

        detail = self._build_detail(request, winner, downgraded, needed_pages, reason)
        return RoutingDecision(
            request_id=request.request_id,
            replica_id=winner.replica.replica_id,
            model=winner.replica.model,
            downgraded=downgraded,
            reason=reason,
            reason_detail=detail,
            score=winner.score,
            score_breakdown=dict(winner.breakdown),
            risk=winner.risk,
            candidates=tuple(e.replica.replica_id for e in evaluations),
        )

    def _should_downgrade(
        self, request: Request, replicas: Sequence[Replica], now_ms: int
    ) -> bool:
        if not request.downgrade_allowed or not request.downgrade_models:
            return False
        best = min(
            (self._estimate_risk(request, r, now_ms) for r in replicas),
            key=lambda risk: risk.level,
        )
        levels = list(RiskLevel)
        return (
            levels.index(best.level)
            >= levels.index(self._config.downgrade_risk_threshold)
        )

    def _downgrade_candidates(
        self, request: Request, needed_pages: int
    ) -> list[Replica]:
        allowed = set(request.downgrade_models)
        return [
            r for r in self._replicas.values()
            if r.is_available
            and r.model in allowed
            and r.model != request.model
            and r.free_pages >= needed_pages
        ]

    def _pick_winner(
        self, evaluations: Sequence[_Evaluation]
    ) -> tuple[_Evaluation, bool]:
        ordered = self._rotate(evaluations)
        winner = ordered[0]
        tie = False
        for candidate in ordered[1:]:
            if math.isclose(candidate.score, winner.score, abs_tol=1e-9):
                tie = True
            elif candidate.score > winner.score:
                winner = candidate
                tie = False
        return winner, tie

    def _rotate(self, evaluations: Sequence[_Evaluation]) -> list[_Evaluation]:
        if not evaluations:
            return []
        offset = self._cursor % len(evaluations)
        return list(evaluations[offset:]) + list(evaluations[:offset])

    def _evaluate(
        self,
        request: Request,
        replica: Replica,
        needed_pages: int,
        now_ms: int,
    ) -> _Evaluation:
        cfg = self._config
        risk = self._estimate_risk(request, replica, now_ms)
        budget = max(1, request.deadline_ms - now_ms)

        urgency = -cfg.weight_queue_risk * min(1.0, risk.estimated_wait_ms / budget)
        load = -cfg.weight_load * min(1.0, replica.load_ratio)
        memory_pressure_ratio = (
            (replica.total_pages - replica.free_pages) / replica.total_pages
            if replica.total_pages > 0
            else 0.0
        )
        memory = -cfg.weight_memory_pressure * memory_pressure_ratio

        hit = float(request.kv_prefix_hits.get(replica.replica_id, 0.0))
        kv = cfg.weight_kv_affinity * hit

        idle_bonus = 0.0
        if replica.queue_depth == 0:
            idle_for = (
                now_ms - replica.last_dispatched_ms
                if replica.last_dispatched_ms is not None
                else cfg.starvation_threshold_ms
            )
            if idle_for >= cfg.starvation_threshold_ms:
                idle_bonus = cfg.weight_starvation
            else:
                idle_bonus = cfg.idle_empty_bonus

        breakdown = {
            "deadline_urgency": urgency,
            "load": load,
            "memory_pressure": memory,
            "kv_affinity": kv,
            "idle_bonus": idle_bonus,
        }
        score = sum(breakdown.values())
        return _Evaluation(replica, score, breakdown, risk)

    def _estimate_service_ms(
        self, request: Request, replica: Replica
    ) -> tuple[float, float]:
        own_prefill = replica.avg_prefill_ms * max(1.0, request.prompt_tokens / 64.0)
        own_decode = replica.avg_decode_per_token_ms * request.max_tokens
        per_queued = replica.avg_prefill_ms + replica.avg_decode_per_token_ms
        wait = replica.queue_depth * per_queued
        return wait, wait + own_prefill + own_decode

    def _estimate_risk(
        self, request: Request, replica: Replica, now_ms: int
    ) -> QueueRisk:
        wait_ms, total_ms = self._estimate_service_ms(request, replica)
        slack_ms = request.deadline_ms - now_ms - total_ms
        budget = max(1, request.deadline_ms - now_ms)

        urgency_ratio = min(1.0, total_ms / budget)
        memory_pressure = (
            (replica.total_pages - replica.free_pages) / replica.total_pages
            if replica.total_pages > 0
            else 0.0
        )
        risk_ratio = max(urgency_ratio, min(1.0, replica.load_ratio), memory_pressure)

        if slack_ms <= 0:
            level = RiskLevel.CRITICAL
        elif risk_ratio >= self._config.critical_risk_ratio:
            level = RiskLevel.CRITICAL
        elif risk_ratio >= self._config.high_risk_ratio:
            level = RiskLevel.HIGH
        elif risk_ratio >= self._config.high_risk_ratio / 2:
            level = RiskLevel.MEDIUM
        elif replica.queue_depth == 0 and replica.load_ratio == 0.0 and memory_pressure == 0.0:
            level = RiskLevel.NONE
        else:
            level = RiskLevel.LOW

        reason = (
            f"队列深度 {replica.queue_depth}，预计排队 {wait_ms:.1f}ms，"
            f"含自身服务共 {total_ms:.1f}ms，deadline 余量 {slack_ms:.1f}ms"
        )
        return QueueRisk(level, wait_ms, slack_ms, reason)

    def _reason(
        self, winner: _Evaluation, tie: bool, downgraded: bool
    ) -> RouteReason:
        if downgraded:
            return RouteReason.DOWNGRADE
        if tie:
            return RouteReason.ROUND_ROBIN
        breakdown = winner.breakdown
        dominant = max(breakdown, key=lambda key: abs(breakdown[key]))
        if dominant == "idle_bonus" and breakdown["idle_bonus"] >= self._config.weight_starvation:
            return RouteReason.IDLE_STARVATION
        if dominant == "kv_affinity":
            hit = breakdown["kv_affinity"] / self._config.weight_kv_affinity
            if hit >= 0.4:
                return RouteReason.KV_AFFINITY
        return RouteReason.LOWEST_LOAD

    def _build_detail(
        self,
        request: Request,
        winner: _Evaluation,
        downgraded: bool,
        needed_pages: int,
        reason: RouteReason,
    ) -> str:
        replica = winner.replica
        hit = request.kv_prefix_hits.get(replica.replica_id, 0.0)
        parts = [
            f"选择副本 {replica.replica_id}（模型 {replica.model}）",
            f"原因 {reason.value}",
            f"队列深度 {replica.queue_depth}",
            f"空闲显存页 {replica.free_pages}/{replica.total_pages}（需 {needed_pages} 页）",
            f"KV 前缀命中率 {hit:.0%}",
            f"综合评分 {winner.score:.3f}",
        ]
        if downgraded:
            parts.append(
                f"已由 {request.model} 降级至允许列表 {list(request.downgrade_models)}"
            )
        return "；".join(parts)
