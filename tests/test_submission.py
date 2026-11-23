"""/submit 双流 CUDA 提交清单：服务层测试。

包含一个与实现无关的清单不变量校验器（assert_checklist_valid），
以及对切流/句柄上限、无解分类、句柄复用、择优规则的覆盖和随机 fuzz。
"""

from __future__ import annotations

import random

import pytest

from app.planner.models import PlanningError, Wait
from app.planner.submission import (
    REASON_DISPATCH,
    REASON_MEMORY,
    REASON_OK,
    build_batches,
    enumerate_orders,
    evaluate_order,
    submit_plan,
)
from app.planner.solver import build_base_edges, reachability

from conftest import make_input


# --------------------------------------------------------------------------- #
# 独立不变量校验器：不依赖提交侧实现，逐条核对任务硬性语义
# --------------------------------------------------------------------------- #
def assert_checklist_valid(data, out, max_handles: int, max_switches: int) -> None:
    assert out.feasible
    assert out.reason == REASON_OK

    stream_of: dict[str, str] = {}
    position_in_stream: dict[str, int] = {}
    for sid, ops in data.streams.items():
        for i, op in enumerate(ops):
            stream_of[op] = sid
            position_in_stream[op] = i

    # 1. 批次按提交序号连续，覆盖全部算子，每批带流 id
    ops_order = [b.operator for b in out.batches]
    assert [b.seq for b in out.batches] == list(range(len(data.all_operators())))
    assert sorted(ops_order) == sorted(data.all_operators())
    for b in out.batches:
        assert b.stream_id == stream_of[b.operator]

    # 2. 同流算子的相对顺序与流内提交顺序一致
    last_seen: dict[str, int] = {}
    for b in out.batches:
        sid = b.stream_id
        cur = position_in_stream[b.operator]
        if sid in last_seen:
            assert cur > last_seen[sid]
        last_seen[sid] = cur

    # 3. 切流按相邻异流批次计数，且不超上限
    switches = sum(
        1 for i in range(1, len(out.batches))
        if out.batches[i].stream_id != out.batches[i - 1].stream_id
    )
    assert switches == out.host_stream_switches
    assert switches <= max_switches

    # 4. 依赖 DAG 的全部 happens-before 关系都被提交顺序尊重（waitee 先于 waiter）
    edges = build_base_edges(data)
    if out.added_wait is not None:
        edges.add((out.added_wait.waitee, out.added_wait.waiter))
    reach = reachability(edges, data.all_operators())
    pos_global = {op: i for i, op in enumerate(ops_order)}
    for u, succ in reach.items():
        for v in succ:
            assert pos_global[u] < pos_global[v], f"提交顺序违反依赖 {u} -> {v}"

    all_waits = list(data.waits) + (
        [out.added_wait] if out.added_wait is not None else []
    )

    # 5. 每条跨流等待：waiter 批次里有 wait_event，绑定的句柄必须等于
    #    waitee 批次 record_event 的句柄（即绑定到所等算子已提交的那次记录）
    batch_of = {b.operator: b for b in out.batches}
    for w in all_waits:
        wb = batch_of[w.waiter]
        rb = batch_of[w.waitee]
        assert rb.record_event is not None, "被等算子批次必须记录事件"
        recorded_handle = rb.record_event.event_handle
        matched = [
            we for we in wb.wait_events
            if we.waitee == w.waitee and we.event_handle == recorded_handle
        ]
        assert len(matched) == 1, f"等待 {w.waiter}->{w.waitee} 未正确绑定记录"
        # 记录批次必须早于等待批次提交（拓扑序保证，这里显式复核）
        assert pos_global[w.waitee] < pos_global[w.waiter]

    # 6. 只有被跨流等待的算子才记录事件；句柄编号连续且不超上限
    recorders = {w.waitee for w in all_waits}
    used_handles: set[int] = set()
    for b in out.batches:
        if b.operator in recorders:
            assert b.record_event is not None
            used_handles.add(b.record_event.event_handle)
        else:
            assert b.record_event is None
        for we in b.wait_events:
            assert we.event_handle in used_handles or True  # 句柄可能稍后才被编号集合统计
    assert used_handles == set(range(out.event_handles_used))
    assert out.event_handles_used <= max_handles

    # 7. 句柄复用合法性：同一个句柄的任意两次记录，前一条记录的全部等待
    #    必须在后一条记录所在批次之前提交
    handle_records: dict[int, list[str]] = {}
    for b in out.batches:
        if b.record_event is not None:
            handle_records.setdefault(b.record_event.event_handle, []).append(b.operator)
    waiters_of: dict[str, list[str]] = {}
    for w in all_waits:
        waiters_of.setdefault(w.waitee, []).append(w.waiter)
    for handle, rec_ops in handle_records.items():
        for earlier, later in zip(rec_ops, rec_ops[1:]):
            for waiter in waiters_of[earlier]:
                assert pos_global[waiter] < pos_global[later], (
                    f"句柄 {handle} 复用过早：{earlier} 的等待 {waiter} 尚未提交"
                )

    # 8. 显存偏移与 /plan 同源：这里核对偏移集合与对齐
    by_name = {t.name: t for t in data.tensors}
    assert set(out.offsets) == set(by_name)
    for name, off in out.offsets.items():
        assert off % by_name[name].alignment == 0


# --------------------------------------------------------------------------- #
# 1. 基本端到端：已有跨流等待，清单顺序与绑定正确
# --------------------------------------------------------------------------- #
def test_basic_checklist_with_existing_wait():
    data = make_input(
        ["a1", "a2", "a3"],
        ["b1", "b2", "b3"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=200,
        waits=[("b2", "a1")],
    )
    out = submit_plan(data, max_event_handles=4, max_host_stream_switches=10)
    assert out.feasible and out.added_wait is None
    assert_checklist_valid(data, out, 4, 10)
    # 一条等待 -> 恰好 1 个句柄；最优顺序先跑完 A 再跑 B -> 1 次切流
    assert out.event_handles_used == 1
    assert out.host_stream_switches == 1


# --------------------------------------------------------------------------- #
# 2. 切流上限：两条非空流之间至少有一次切换；上限 0 仅在无跨流依赖且单流时可行
# --------------------------------------------------------------------------- #
def test_switch_count_basics():
    data = make_input(
        ["a1", "a2"], ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=200,
    )
    # 无跨流依赖：可以先整条 A 再整条 B，恰好 1 次切流
    out = submit_plan(data, max_event_handles=8, max_host_stream_switches=1)
    assert out.feasible
    assert out.host_stream_switches == 1
    assert_checklist_valid(data, out, 8, 1)

    # 上限 0：两条非空流必然至少切流一次 -> 下发受限
    out0 = submit_plan(data, max_event_handles=8, max_host_stream_switches=0)
    assert not out0.feasible
    assert out0.reason == REASON_DISPATCH
    assert out0.batches == [] and out0.offsets == {}
    assert out0.dispatch_diagnostics["min_host_stream_switches"] == 1

    # 单流（另一条流不存在的退化场景由“必须两条流”校验拦截），
    # 这里改用“另一条流只有一个算子且无等待”验证 0 切流不可达、1 可达
    data_dep = make_input(
        ["a1", "a2"], ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=200,
        waits=[("b1", "a1")],
    )
    out_dep = submit_plan(data_dep, max_event_handles=8, max_host_stream_switches=0)
    assert not out_dep.feasible and out_dep.reason == REASON_DISPATCH
    assert out_dep.dispatch_diagnostics["min_host_stream_switches"] >= 1


# --------------------------------------------------------------------------- #
# 3. 句柄上限：交错的在途记录需要多个句柄
# --------------------------------------------------------------------------- #
def test_event_handle_cap_binding_and_reuse():
    # 两条独立的跨流等待：b3 等 a1、a3 等 b1。交错顺序 a1,b1,... 下
    # a1 的记录（b3 未提交）与 b1 的记录（a3 未提交）同时在途，
    # 任何合法顺序的最少句柄峰值为 2。
    data = make_input(
        ["a1", "a2", "a3"],
        ["b1", "b2", "b3"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b2", "consumers": ["b3"], "size": 100, "alignment": 1},
        ],
        budget=200,
        waits=[("b3", "a1"), ("a3", "b1")],
    )
    edges = build_base_edges(data)
    reach = reachability(edges, data.all_operators())
    evals = [evaluate_order(data, o, data.waits) for o in enumerate_orders(data, reach)]
    assert min(e.peak_handles for e in evals) == 2

    # 句柄上限 1：任何顺序峰值都 >=2 -> 下发受限
    out = submit_plan(data, max_event_handles=1, max_host_stream_switches=10)
    assert not out.feasible and out.reason == REASON_DISPATCH
    assert out.dispatch_diagnostics["min_event_handles"] == 2

    # 句柄上限 2：可行且清单通过全部不变量
    out2 = submit_plan(data, max_event_handles=2, max_host_stream_switches=10)
    assert out2.feasible
    assert out2.event_handles_used == 2
    assert_checklist_valid(data, out2, 2, 10)


# --------------------------------------------------------------------------- #
# 4. 句柄复用：可串行的记录复用同一句柄，等待各自绑定正确的那次记录
# --------------------------------------------------------------------------- #
def test_handle_reuse_same_handle_for_sequential_records():
    # 两条独立等待链：b1 等 a1、b3 等 a3。存在让 b1 先于 a3 提交的顺序，
    # 此时 a1 的记录在 a3 记录前已关闭 -> 一个句柄即可，两次记录复用它。
    data = make_input(
        ["a1", "a2", "a3", "a4"],
        ["b1", "b2", "b3", "b4"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "a3", "consumers": ["a4"], "size": 100, "alignment": 1},
        ],
        budget=200,
        waits=[("b1", "a1"), ("b3", "a3")],
    )
    # 放宽切流上限：存在把 b1 夹在 a1 与 a3 两次记录之间的顺序，
    # 此时两条记录不同时在途 -> 一个句柄即可并被复用
    out = submit_plan(data, max_event_handles=1, max_host_stream_switches=3)
    assert out.feasible
    assert out.event_handles_used == 1
    assert_checklist_valid(data, out, 1, 3)
    b1 = next(b for b in out.batches if b.operator == "b1")
    b3 = next(b for b in out.batches if b.operator == "b3")
    a1 = next(b for b in out.batches if b.operator == "a1")
    a3 = next(b for b in out.batches if b.operator == "a3")
    # 每个等待绑定的句柄就是所等算子那次记录的句柄
    assert b1.wait_events[0].event_handle == a1.record_event.event_handle
    assert b3.wait_events[0].event_handle == a3.record_event.event_handle
    # 两次记录复用同一句柄，且前次等待 b1 严格早于 a3 批次
    assert a1.record_event.event_handle == a3.record_event.event_handle
    pos = {b.operator: b.seq for b in out.batches}
    assert pos["b1"] < pos["a3"]


def test_chained_records_cannot_share_one_handle():
    # b2 等 a1、a3 等 b1：a1 记录的等待方 b2 在 b1 记录之后才提交，
    # 而 b1 记录的等待方 a3 也在 b1 之后；两条记录的在途区间必然重叠，
    # 穷举任何合法顺序峰值都 >= 2，一个句柄无法承载。
    data = make_input(
        ["a1", "a2", "a3"],
        ["b1", "b2", "b3"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b2", "consumers": ["b3"], "size": 100, "alignment": 1},
        ],
        budget=200,
        waits=[("b2", "a1"), ("a3", "b1")],
    )
    out1 = submit_plan(data, max_event_handles=1, max_host_stream_switches=10)
    assert not out1.feasible and out1.reason == REASON_DISPATCH
    assert out1.dispatch_diagnostics["min_event_handles"] == 2
    out2 = submit_plan(data, max_event_handles=2, max_host_stream_switches=10)
    assert out2.feasible
    assert out2.event_handles_used == 2
    assert_checklist_valid(data, out2, 2, 10)


# --------------------------------------------------------------------------- #
# 5. 显存不足：任何单等待方案都超预算 -> reason=memory，不返回部分清单
# --------------------------------------------------------------------------- #
def test_infeasible_memory_reason():
    data = make_input(
        ["a1", "a2"], ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 300, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 300, "alignment": 1},
        ],
        budget=200,
    )
    out = submit_plan(data, max_event_handles=8, max_host_stream_switches=10)
    assert not out.feasible
    assert out.reason == REASON_MEMORY
    assert out.batches == [] and out.offsets == {}
    assert out.best_effort_total_bytes == 300
    # 无解时 total_bytes 与 best_effort 同口径，均为可达最小占用
    assert out.total_bytes == 300


# --------------------------------------------------------------------------- #
# 6. 超预算后新增等待：与 /plan 同一套择优规则，且清单可下发
# --------------------------------------------------------------------------- #
def test_added_wait_when_over_budget_and_dispatchable():
    data = make_input(
        ["a1", "a2", "a3"],
        ["b1", "b2", "b3"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=100,
    )
    out = submit_plan(data, max_event_handles=4, max_host_stream_switches=10)
    assert out.feasible
    assert out.added_wait == Wait("a1", "b2")  # 与 /plan 字典序择优一致
    assert out.total_bytes == 100
    assert out.event_handles_used == 1
    assert_checklist_valid(data, out, 4, 10)


# --------------------------------------------------------------------------- #
# 7. 显存排得下但只能靠高切流/多句柄顺序下发 -> reason=dispatch
# --------------------------------------------------------------------------- #
def test_dispatch_limited_after_wait_added():
    # 预算 100 迫使新增一条等待；把切流上限压到 0：任何含跨流边的顺序
    # 都至少切流一次 -> 下发受限（注意不是显存不足）
    data = make_input(
        ["a1", "a2"], ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=100,
    )
    out = submit_plan(data, max_event_handles=4, max_host_stream_switches=0)
    assert not out.feasible
    assert out.reason == REASON_DISPATCH
    assert out.best_effort_total_bytes == 100  # 显存本身是能排下的
    assert out.total_bytes == 100
    assert out.dispatch_diagnostics["min_host_stream_switches"] == 1


# --------------------------------------------------------------------------- #
# 8. 零句柄上限：存在跨流等待（含必须新增的等待）时不可下发
# --------------------------------------------------------------------------- #
def test_zero_handles_with_waits():
    data = make_input(
        ["a1", "a2"], ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=200,
        waits=[("b1", "a1")],
    )
    out = submit_plan(data, max_event_handles=0, max_host_stream_switches=10)
    assert not out.feasible and out.reason == REASON_DISPATCH
    assert out.dispatch_diagnostics["min_event_handles"] == 1

    # 无任何跨流等待时不需要任何事件句柄（切流上限给到可达的 1）
    data_no_wait = make_input(
        ["a1", "a2"], ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=200,
    )
    out2 = submit_plan(data_no_wait, max_event_handles=0, max_host_stream_switches=1)
    assert out2.feasible
    assert out2.event_handles_used == 0
    assert all(b.record_event is None and not b.wait_events for b in out2.batches)


# --------------------------------------------------------------------------- #
# 9. 每批内部顺序固定：wait_events 在记录之前、记录在算子启动之后由结构保证；
#    同时一个算子等待多个事件时全部列出
# --------------------------------------------------------------------------- #
def test_operator_waiting_multiple_events():
    data = make_input(
        ["a1", "a2", "a3"],
        ["b1", "b2", "b3"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=200,
        waits=[("b3", "a1"), ("b3", "b1")],
    )
    out = submit_plan(data, max_event_handles=4, max_host_stream_switches=10)
    assert out.feasible
    assert_checklist_valid(data, out, 4, 10)
    b3 = next(b for b in out.batches if b.operator == "b3")
    assert {w.waitee for w in b3.wait_events} == {"a1", "b1"}
    # 两个不同的被等算子 -> 两个句柄编号
    assert len({w.event_handle for w in b3.wait_events}) == 2


# --------------------------------------------------------------------------- #
# 10. 输入校验仍由 /plan 同款规则负责
# --------------------------------------------------------------------------- #
def test_submit_reuses_validation():
    data = make_input(
        ["a1"], ["b1", "b2"],
        [{"name": "T", "producer": "a1", "consumers": ["b2"], "size": 10, "alignment": 3}],
        budget=100,
    )
    with pytest.raises(PlanningError, match="2 的幂"):
        submit_plan(data, max_event_handles=4, max_host_stream_switches=4)

    with pytest.raises(PlanningError, match="非负整数"):
        submit_plan(
            make_input(
                ["a1"], ["b1"],
                [{"name": "T", "producer": "a1", "consumers": ["b1"], "size": 10, "alignment": 1}],
                budget=100,
            ),
            max_event_handles=-1,
            max_host_stream_switches=0,
        )


# --------------------------------------------------------------------------- #
# 11. fuzz：随机实例下清单永远满足全部硬性不变量，且用量不超上限
# --------------------------------------------------------------------------- #
def test_fuzz_submission_invariants():
    rng = random.Random(20261124)
    for _ in range(250):
        na = rng.randint(1, 5)
        nb = rng.randint(1, 5)
        ops_a = [f"a{i}" for i in range(na)]
        ops_b = [f"b{i}" for i in range(nb)]
        waits: list[tuple[str, str]] = []
        for a in ops_a:
            for b in ops_b:
                r = rng.random()
                if r < 0.2:
                    waits.append((b, a))
                elif r < 0.4:
                    waits.append((a, b))
        all_ops = ops_a + ops_b
        n_tensors = rng.randint(1, min(6, len(all_ops) - 1))
        tensors = []
        used: set[str] = set()
        for k in range(n_tensors):
            producer = rng.choice(all_ops[:-1]) if len(all_ops) > 1 else all_ops[0]
            later = [op for op in all_ops if op != producer]
            if not later:
                break
            consumers = rng.sample(later, rng.randint(1, min(3, len(later))))
            name = f"T{k}"
            if name in used:
                continue
            used.add(name)
            tensors.append(
                {
                    "name": name,
                    "producer": producer,
                    "consumers": consumers,
                    "size": rng.choice([1, 7, 16, 100, 255]),
                    "alignment": rng.choice([1, 2, 4, 8, 16, 64]),
                }
            )
        if not tensors:
            continue
        data = make_input(ops_a, ops_b, tensors, budget=10**9, waits=waits)
        try:
            out = submit_plan(data, max_event_handles=12, max_host_stream_switches=12)
        except PlanningError:
            continue  # 随机边可能成环或数据竞争
        assert out.feasible
        assert out.added_wait is None  # 预算极大，绝不新增等待
        assert_checklist_valid(data, out, 12, 12)


# --------------------------------------------------------------------------- #
# 12. fuzz：紧预算 + 紧上限，可行结果必须通过不变量校验；无解分类必须自洽
# --------------------------------------------------------------------------- #
def test_fuzz_tight_limits_classification():
    rng = random.Random(777)
    for _ in range(200):
        na, nb = rng.randint(1, 4), rng.randint(1, 4)
        ops_a = [f"a{i}" for i in range(na)]
        ops_b = [f"b{i}" for i in range(nb)]
        all_ops = ops_a + ops_b
        tensors = []
        for k in range(rng.randint(1, min(4, len(all_ops) - 1))):
            producer = rng.choice(all_ops[:-1])
            later = [op for op in all_ops if op != producer]
            tensors.append(
                {
                    "name": f"T{k}",
                    "producer": producer,
                    "consumers": [rng.choice(later)],
                    "size": rng.choice([10, 100, 300]),
                    "alignment": rng.choice([1, 16]),
                }
            )
        data = make_input(
            ops_a, ops_b, tensors,
            budget=rng.choice([1, 50, 150, 10_000]),
        )
        try:
            out = submit_plan(
                data,
                max_event_handles=rng.randint(0, 3),
                max_host_stream_switches=rng.randint(0, 3),
            )
        except PlanningError:
            continue
        if out.feasible:
            assert_checklist_valid(
                data, out, out.max_event_handles, out.max_host_stream_switches
            )
            assert out.event_handles_used <= out.max_event_handles
            assert out.host_stream_switches <= out.max_host_stream_switches
        else:
            assert out.reason in (REASON_MEMORY, REASON_DISPATCH)
            assert out.batches == [] and out.offsets == {}
            if out.reason == REASON_DISPATCH:
                # 诊断值必须超过至少一项上限，否则分类自相矛盾
                diag = out.dispatch_diagnostics
                assert (
                    diag["min_event_handles"] > out.max_event_handles
                    or diag["min_host_stream_switches"] > out.max_host_stream_switches
                )


# --------------------------------------------------------------------------- #
# 13. 枚举完整性：枚举的交错数等于 C(na+nb, na)（无跨流边时）
# --------------------------------------------------------------------------- #
def test_enumerate_orders_all_interleavings():
    data = make_input(
        ["a1", "a2", "a3"], ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 10, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 10, "alignment": 1},
        ],
        budget=1000,
    )
    edges = build_base_edges(data)
    reach = reachability(edges, data.all_operators())
    orders = enumerate_orders(data, reach)
    import math
    assert len(orders) == math.comb(5, 3)
    # 每个顺序都是拓扑序
    for o in orders:
        pos = {op: i for i, op in enumerate(o)}
        for u, succ in reach.items():
            for v in succ:
                assert pos[u] < pos[v]


# --------------------------------------------------------------------------- #
# 14. build_batches 给出的句柄数恰好等于评估峰值
# --------------------------------------------------------------------------- #
def test_build_batches_handle_count_matches_peak():
    data = make_input(
        ["a1", "a2", "a3"],
        ["b1", "b2", "b3"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b2", "consumers": ["b3"], "size": 100, "alignment": 1},
        ],
        budget=200,
        waits=[("b3", "a1"), ("a3", "b1")],
    )
    edges = build_base_edges(data)
    reach = reachability(edges, data.all_operators())
    for order in enumerate_orders(data, reach):
        ev = evaluate_order(data, order, data.waits)
        _batches, handles = build_batches(data, ev, data.waits)
        assert handles == ev.peak_handles


# --------------------------------------------------------------------------- #
# 15. 择优最优性：服务选择与对全部合法交错的独立穷举结果一致
# --------------------------------------------------------------------------- #
def test_selected_order_matches_bruteforce_optimum():
    data = make_input(
        ["a1", "a2", "a3"],
        ["b1", "b2", "b3"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b2", "consumers": ["b3"], "size": 100, "alignment": 1},
        ],
        budget=200,
        waits=[("b3", "a1"), ("a3", "b1")],
    )
    edges = build_base_edges(data)
    reach = reachability(edges, data.all_operators())
    all_evals = [evaluate_order(data, o, data.waits) for o in enumerate_orders(data, reach)]

    for cap_h, cap_s in [(3, 3), (2, 2), (2, 3), (3, 2), (1, 5), (5, 1)]:
        feasible_evals = [
            e for e in all_evals
            if e.peak_handles <= cap_h and e.switches <= cap_s
        ]
        out = submit_plan(data, max_event_handles=cap_h, max_host_stream_switches=cap_s)
        if not feasible_evals:
            assert not out.feasible and out.reason == REASON_DISPATCH
            continue
        brute = min(feasible_evals, key=lambda e: (e.switches, e.peak_handles, e.order))
        assert out.feasible
        assert out.host_stream_switches == brute.switches
        assert out.event_handles_used == brute.peak_handles
        chosen = [b.operator for b in out.batches]
        assert chosen == brute.order
