# inference-request-router

纯标准库实现的 Python 推理请求路由器。根据请求 deadline、模型版本、最大
token、副本队列、显存页、KV 前缀命中率与允许降级模型，为每条请求选择目标
副本，并输出**选择原因**与**预计排队风险**。

## 解决的问题

- **多副本 round-robin 盲目分发**：连续对话（同一前缀）落到没有 KV Cache
  的副本，重复 prefill 浪费算力并抬高首 token 延迟。
- **大模型排长队、小模型闲置**：部分请求本可降级到小模型，但路由层不会
  根据排队风险主动切换。
- **亲和性与公平性冲突**：只看 KV 命中率会让部分副本过载、空闲副本饥饿；
  只做轮询又丢失缓存收益。本路由器用加权评分 + 饥饿保护同时约束两者。

## 快速开始

```python
from router import InferenceRouter, Replica, Request

replicas = [
    Replica("a", "llm-v2", queue_depth=8, total_pages=128, free_pages=90),
    Replica("b", "llm-v2", queue_depth=0, total_pages=128, free_pages=128),
    Replica("c", "llm-small", queue_depth=1, total_pages=128, free_pages=100),
]
router = InferenceRouter(replicas)

decision = router.route(Request(
    request_id="chat-77",
    model="llm-v2",
    max_tokens=200,
    prompt_tokens=800,
    deadline_ms=10_000,
    kv_prefix_hits={"a": 0.85},
    downgrade_allowed=True,
    downgrade_models=("llm-small",),
))

print(decision.replica_id, decision.reason.value, decision.risk.level.value)
print(decision.reason_detail)
print(decision.score_breakdown)
```

`RoutingDecision` 包含目标副本、是否降级、原因码（`round_robin` /
`lowest_load` / `kv_affinity` / `idle_starvation` / `downgrade`）、分项评分、
排队风险（等级、预计等待毫秒、deadline 余量）以及全部候选副本。

## 路由流程

1. **可行性过滤**：只保留 `ONLINE` 副本（`DRAINING`/`OFFLINE` 不接新流量），
   模型版本必须精确匹配；显存页按 `ceil((prompt_tokens + max_tokens) /
   page_size_tokens)` 计算，页数不足的副本剔除。
2. **降级判断**：主模型无可行副本，或最优主副本风险达到
   `downgrade_risk_threshold`（默认 HIGH）时，才在允许列表内挑选小模型；
   禁用降级时即使排队严重也留在主模型。降级不能替代版本匹配。
3. **打分排序**：候选副本按综合分降序，同分由轮转游标做 round-robin。
4. **状态回写**：记录中选副本的派发时间，供后续饥饿判定使用。

## 评分模型

综合分由五项相加（惩罚项为负、奖励项为正）：

| 分项 | 方向 | 作用 |
| --- | --- | --- |
| `deadline_urgency` | 负 | 预计等待占 deadline 预算比例越高，扣分越多 |
| `load` | 负 | 已占用显存页比例，刻画副本整体负载 |
| `memory_pressure` | 负 | 显存压力，避免把请求塞进接近满载的副本 |
| `kv_affinity` | 正 | KV 前缀命中率 × 权重，命中率高的副本优先 |
| `idle_bonus` | 正 | 空队列副本的饥饿保护（见下） |

所有权重与阈值集中在 `RouterConfig`，可按业务调参，无需改路由代码。

## 公平性与亲和性的权衡

这是本路由器的核心设计点，二者通过两个机制显式平衡，而不是让单一指标
决定一切：

- **亲和性（避免重复 prefill）**：`kv_affinity` 给持有前缀缓存的副本加权。
  当命中副本的队列惩罚小于缓存收益时，连续对话会稳定落在同一副本上
  （测试 `test_continuous_conversation_stays_on_warm_replica` 覆盖）。
- **公平性（空闲副本不饥饿）**：空队列副本获得 `idle_bonus`。其中空闲
  超过 `starvation_threshold_ms`（默认 2s）且仍未被选中的副本直接拿到
  满分饥饿奖励 `weight_starvation`，可以压过 KV 亲和分，强制把流量引过去；
  未达阈值的空副本只拿较小的常数奖励，避免刚空闲就频繁抢占。
- **同分轮询**：完全对等的副本之间由游标 round-robin 分发，突发并发下
  各副本收到的请求数最多相差 1（测试
  `test_burst_round_robin_across_identical_replicas` 覆盖）。
- **可调的平衡点**：调大 `weight_kv_affinity` 更偏缓存亲和（省算力、降低
  TTFT），调大 `weight_starvation` / 调小 `starvation_threshold_ms` 更偏
  负载均衡。默认参数下，亲和性可以克服中等排队惩罚，但无法长期压住一个
  真正空闲且饥饿的副本。

注意：饥饿保护只针对**空队列**副本。一个仍在处理请求的副本不会因为“相对
较闲”就抢占高命中副本，避免缓存被无谓打散。

## 排队风险

`QueueRisk` 综合三个信号给出 `NONE/LOW/MEDIUM/HIGH/CRITICAL`：

- 预计等待 = `queue_depth × (平均 prefill + 平均单 token decode)`；
- 再叠加本请求自身服务时间得到总耗时，与 `deadline_ms` 比较得到余量
  `slack_ms`；余量 ≤ 0 直接判 `CRITICAL`；
- 同时参考队列占用与显存压力比例，防止“队列短但显存即将耗尽”被低估。

风险等级同时用于降级触发与 `RoutingDecision.reason` 解释。

## 测试

覆盖长上下文分页、并发突发轮询、副本下线/上线、版本不匹配、降级启用与
禁用、KV 亲和、饥饿保护与 deadline 风险：

```bash
python -m unittest
```

共 21 个用例，全部基于依赖注入的假时钟，饥饿相关时序测试确定性可复现。

## 模块结构

- `router/models.py`：`Request` / `Replica` / `RouterConfig` /
  `RoutingDecision` / `QueueRisk` 等数据模型与枚举。
- `router/router.py`：`InferenceRouter` 评分路由、降级、风险估算与异常类型。
- `tests/test_router.py`：单元测试。
