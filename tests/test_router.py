import unittest

from router import (
    IncompatibleModelError,
    InferenceRouter,
    InsufficientCapacityError,
    NoReplicaAvailableError,
    Replica,
    ReplicaStatus,
    Request,
    RiskLevel,
    RouteReason,
    RouterConfig,
)


class FakeClock:
    def __init__(self, start: int = 0) -> None:
        self.now = start

    def __call__(self) -> int:
        return self.now


def make_replica(
    replica_id: str,
    model: str = "llm-large",
    *,
    queue_depth: int = 0,
    total_pages: int = 100,
    free_pages: int = 100,
    status: ReplicaStatus = ReplicaStatus.ONLINE,
    last_dispatched_ms: int | None = None,
) -> Replica:
    return Replica(
        replica_id=replica_id,
        model=model,
        status=status,
        queue_depth=queue_depth,
        total_pages=total_pages,
        free_pages=free_pages,
        last_dispatched_ms=last_dispatched_ms,
    )


def make_request(
    request_id: str = "req-1",
    *,
    model: str = "llm-large",
    max_tokens: int = 32,
    prompt_tokens: int = 0,
    deadline_ms: int = 10_000,
    kv_prefix_hits=None,
    downgrade_allowed: bool = False,
    downgrade_models=(),
) -> Request:
    return Request(
        request_id=request_id,
        model=model,
        max_tokens=max_tokens,
        prompt_tokens=prompt_tokens,
        deadline_ms=deadline_ms,
        kv_prefix_hits=kv_prefix_hits or {},
        downgrade_allowed=downgrade_allowed,
        downgrade_models=tuple(downgrade_models),
    )


class PageSizingTest(unittest.TestCase):
    def test_long_context_pages_round_up(self) -> None:
        clock = FakeClock()
        router = InferenceRouter([make_replica("r1")], clock_ms=clock)
        request = make_request(prompt_tokens=1000, max_tokens=100)
        # ceil(1100 / 16) == 69
        self.assertEqual(router.required_pages(request), 69)

    def test_minimum_one_page(self) -> None:
        router = InferenceRouter([make_replica("r1")], clock_ms=FakeClock())
        self.assertEqual(router.required_pages(make_request(max_tokens=1)), 1)

    def test_replica_without_enough_pages_is_skipped(self) -> None:
        clock = FakeClock()
        small = make_replica("small", total_pages=100, free_pages=50)
        roomy = make_replica("roomy", total_pages=200, free_pages=120)
        router = InferenceRouter([small, roomy], clock_ms=clock)
        decision = router.route(make_request(prompt_tokens=1000, max_tokens=100))
        self.assertEqual(decision.replica_id, "roomy")

    def test_insufficient_capacity_raises(self) -> None:
        router = InferenceRouter(
            [make_replica("r1", total_pages=10, free_pages=5)],
            clock_ms=FakeClock(),
        )
        with self.assertRaises(InsufficientCapacityError):
            router.route(make_request(prompt_tokens=1000, max_tokens=100))


class BurstFairnessTest(unittest.TestCase):
    def test_burst_round_robin_across_identical_replicas(self) -> None:
        clock = FakeClock()
        replicas = [make_replica("r1"), make_replica("r2"), make_replica("r3")]
        router = InferenceRouter(replicas, clock_ms=clock)
        chosen = [
            router.route(make_request(f"req-{i}")).replica_id for i in range(9)
        ]
        counts = {rid: chosen.count(rid) for rid in ("r1", "r2", "r3")}
        self.assertEqual(set(counts.values()), {3})
        self.assertEqual(chosen[:3], ["r1", "r2", "r3"])

    def test_decision_marks_round_robin_reason(self) -> None:
        router = InferenceRouter(
            [make_replica("r1"), make_replica("r2")], clock_ms=FakeClock()
        )
        decision = router.route(make_request())
        self.assertEqual(decision.reason, RouteReason.ROUND_ROBIN)

    def test_duplicate_replica_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InferenceRouter([make_replica("dup"), make_replica("dup")])


class KVAffinityTest(unittest.TestCase):
    def test_affinity_overcomes_moderate_queue_penalty(self) -> None:
        clock = FakeClock()
        empty = make_replica("empty", queue_depth=0, free_pages=80)
        warm = make_replica("warm", queue_depth=2, free_pages=60)
        router = InferenceRouter([empty, warm], clock_ms=clock)
        decision = router.route(
            make_request(kv_prefix_hits={"empty": 0.0, "warm": 0.9})
        )
        self.assertEqual(decision.replica_id, "warm")
        self.assertEqual(decision.reason, RouteReason.KV_AFFINITY)

    def test_idle_replica_not_starved_forever(self) -> None:
        clock = FakeClock()
        idle = make_replica("idle", queue_depth=0, last_dispatched_ms=0)
        warm = make_replica(
            "warm", queue_depth=2, free_pages=60, last_dispatched_ms=0
        )
        router = InferenceRouter([idle, warm], clock_ms=clock)
        first = router.route(
            make_request("req-1", kv_prefix_hits={"warm": 0.9})
        )
        self.assertEqual(first.replica_id, "warm")

        clock.now = 2_001
        second = router.route(
            make_request("req-2", kv_prefix_hits={"warm": 0.9})
        )
        self.assertEqual(second.replica_id, "idle")
        self.assertEqual(second.reason, RouteReason.IDLE_STARVATION)

    def test_continuous_conversation_stays_on_warm_replica(self) -> None:
        clock = FakeClock()
        replicas = [make_replica("r1"), make_replica("r2")]
        router = InferenceRouter(replicas, clock_ms=clock)
        for i in range(3):
            decision = router.route(
                make_request(f"chat-{i}", kv_prefix_hits={"r2": 0.8})
            )
            self.assertEqual(decision.replica_id, "r2")


class ReplicaLifecycleTest(unittest.TestCase):
    def test_offline_and_draining_replicas_excluded(self) -> None:
        clock = FakeClock()
        online = make_replica("online")
        draining = make_replica("draining", status=ReplicaStatus.DRAINING)
        offline = make_replica("offline", status=ReplicaStatus.OFFLINE)
        router = InferenceRouter([online, draining, offline], clock_ms=clock)
        for i in range(3):
            decision = router.route(make_request(f"req-{i}"))
            self.assertEqual(decision.replica_id, "online")

    def test_all_replicas_offline_raises(self) -> None:
        router = InferenceRouter(
            [make_replica("r1", status=ReplicaStatus.OFFLINE)],
            clock_ms=FakeClock(),
        )
        with self.assertRaises(NoReplicaAvailableError):
            router.route(make_request())

    def test_replica_taken_offline_via_update(self) -> None:
        clock = FakeClock()
        r1 = make_replica("r1")
        r2 = make_replica("r2")
        router = InferenceRouter([r1, r2], clock_ms=clock)
        self.assertEqual(router.route(make_request("a")).replica_id, "r1")
        router.update_replica(
            make_replica("r1", status=ReplicaStatus.OFFLINE)
        )
        decision = router.route(make_request("b"))
        self.assertEqual(decision.replica_id, "r2")


class VersionMismatchTest(unittest.TestCase):
    def test_unsupported_model_version_raises(self) -> None:
        router = InferenceRouter(
            [make_replica("r1", model="llm-v1")], clock_ms=FakeClock()
        )
        with self.assertRaises(IncompatibleModelError):
            router.route(make_request(model="llm-v2"))

    def test_downgrade_cannot_substitute_for_version_match(self) -> None:
        router = InferenceRouter(
            [make_replica("r1", model="small-model")], clock_ms=FakeClock()
        )
        request = make_request(
            model="missing-large",
            downgrade_allowed=True,
            downgrade_models=["small-model"],
        )
        with self.assertRaises(IncompatibleModelError):
            router.route(request)


class DowngradeTest(unittest.TestCase):
    def _pressure_router(self, clock: FakeClock) -> InferenceRouter:
        primary = make_replica(
            "large", model="llm-large", queue_depth=16, free_pages=90
        )
        small_a = make_replica(
            "small-a", model="llm-small", queue_depth=0, free_pages=80
        )
        small_b = make_replica(
            "small-b", model="llm-small", queue_depth=0, free_pages=70
        )
        return InferenceRouter([primary, small_a, small_b], clock_ms=clock)

    def test_downgrade_when_primary_risk_high(self) -> None:
        clock = FakeClock()
        router = self._pressure_router(clock)
        request = make_request(
            "req",
            deadline_ms=500,
            downgrade_allowed=True,
            downgrade_models=["llm-small"],
        )
        decision = router.route(request)
        self.assertTrue(decision.downgraded)
        self.assertEqual(decision.model, "llm-small")
        self.assertEqual(decision.reason, RouteReason.DOWNGRADE)
        self.assertIn("small", decision.replica_id)

    def test_downgrade_disabled_stays_on_primary(self) -> None:
        clock = FakeClock()
        router = self._pressure_router(clock)
        decision = router.route(make_request("req", deadline_ms=500))
        self.assertFalse(decision.downgraded)
        self.assertEqual(decision.replica_id, "large")
        self.assertIn(decision.risk.level, (RiskLevel.HIGH, RiskLevel.CRITICAL))

    def test_downgrade_allowed_but_healthy_primary_kept(self) -> None:
        clock = FakeClock()
        healthy_primary = make_replica("large", model="llm-large")
        small = make_replica("small-a", model="llm-small")
        router = InferenceRouter([healthy_primary, small], clock_ms=clock)
        request = make_request(
            "req",
            downgrade_allowed=True,
            downgrade_models=["llm-small"],
        )
        decision = router.route(request)
        self.assertFalse(decision.downgraded)
        self.assertEqual(decision.replica_id, "large")


class RiskReportingTest(unittest.TestCase):
    def test_risk_fields_populated(self) -> None:
        clock = FakeClock()
        replica = make_replica("r1", queue_depth=4, free_pages=80)
        router = InferenceRouter([replica], clock_ms=clock)
        decision = router.route(make_request(deadline_ms=100_000))
        risk = decision.risk
        self.assertIsInstance(risk.level, RiskLevel)
        self.assertGreater(risk.estimated_wait_ms, 0)
        self.assertTrue(risk.deadline_met)
        self.assertIn("队列深度", risk.reason)
        self.assertEqual(
            set(decision.score_breakdown),
            {
                "deadline_urgency",
                "load",
                "memory_pressure",
                "kv_affinity",
                "idle_bonus",
            },
        )

    def test_expired_deadline_is_critical(self) -> None:
        clock = FakeClock(start=5_000)
        router = InferenceRouter([make_replica("r1")], clock_ms=clock)
        decision = router.route(make_request(deadline_ms=1_000))
        self.assertEqual(decision.risk.level, RiskLevel.CRITICAL)
        self.assertFalse(decision.risk.deadline_met)

    def test_idle_replica_reports_none_risk(self) -> None:
        router = InferenceRouter([make_replica("r1")], clock_ms=FakeClock())
        decision = router.route(make_request(deadline_ms=10_000))
        self.assertEqual(decision.risk.level, RiskLevel.NONE)


if __name__ == "__main__":
    unittest.main()
