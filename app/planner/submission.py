"""双流推理的 CUDA 主机提交清单生成（只依赖标准库）。

在 /plan 显存布局之上，回答“主机如何按顺序把算子下发到两条流”：

- 每个算子一个批次，批次带流 id，按主机提交顺序排列；同一条流的批次相对顺序
  固定为流内提交顺序，两条流之间可交错。
- 合法的主机提交顺序即依赖 DAG（流内顺序边 + 跨流等待边）的拓扑序。
- 每批在同一流里先提交 cudaStreamWaitEvent（本算子的跨流等待），再启动算子，
  随后按需 cudaEventRecord（本算子被其他流等待时记录一次事件）。
- 跨流等待必须绑定“所等算子已提交的那次记录”：只有 waitee 批次早于 waiter
  批次提交时该等待才可下发，而这正是依赖边 waitee->waiter 对拓扑序的要求。
- 主机全程不同步 GPU；设备依赖只来自流内顺序与方案中的等待。
- 事件句柄可复用：一条记录在“最后一个等待它的批次提交之后”即可回收句柄。
  对固定提交顺序，最少句柄数等于逐前缀“已记录且等待未全部提交”记录数的峰值，
  贪心区间着色即可给出确定性的句柄编号。
- 切流次数 = 相邻批次属于不同流的次数。
"""

from __future__ import annotations

from .models import (
    InterleavingEval,
    PlanningError,
    PlanningInput,
    RecordEventCall,
    SubmissionBatch,
    SubmitOutcome,
    Wait,
    WaitEventCall,
)
from .solver import (
    build_base_edges,
    build_interference,
    candidate_waits,
    reachability,
    solve_allocation,
    validate_input,
    validate_tensor_lifecycles,
)

REASON_OK = "ok"
REASON_MEMORY = "memory"
REASON_DISPATCH = "dispatch"


# --------------------------------------------------------------------------- #
# 单条依赖图下的显存布局（沿用 /plan 的布局算法）
# --------------------------------------------------------------------------- #
def _allocate(
    data: PlanningInput, edges: set[tuple[str, str]]
):
    nodes = data.all_operators()
    reach = reachability(edges, nodes)
    validate_tensor_lifecycles(data, reach)
    interfere = build_interference(data, reach)
    return solve_allocation(data, interfere), reach


# --------------------------------------------------------------------------- #
# 合法主机提交顺序：两流固定序列在跨流偏序约束下的全部交错（拓扑序）
# --------------------------------------------------------------------------- #
def enumerate_orders(
    data: PlanningInput,
    reach: dict[str, frozenset[str]],
) -> list[list[str]]:
    """枚举全部合法提交顺序。

    每条流内部顺序固定，因此任意提交顺序都是两个流序列的一个交错；
    跨流等待边 waitee->waiter 要求 waitee 先出现。DFS 时按流 id 字典序
    选择先推进哪条流，保证结果确定。
    """
    stream_ids = sorted(data.streams)
    seq = {sid: list(data.streams[sid]) for sid in stream_ids}
    pos = {sid: 0 for sid in stream_ids}
    placed: set[str] = set()
    results: list[list[str]] = []
    prefix: list[str] = []
    total = sum(len(v) for v in seq.values())

    def dfs() -> None:
        if len(prefix) == total:
            results.append(list(prefix))
            return
        for sid in stream_ids:
            i = pos[sid]
            if i >= len(seq[sid]):
                continue
            op = seq[sid][i]
            # op 的全部跨流前驱都必须已经提交（流内前驱由序列推进天然保证）
            if any(
                pred not in placed and _stream_of(data, pred) != sid
                for pred in _predecessors(reach, op)
            ):
                continue
            prefix.append(op)
            placed.add(op)
            pos[sid] += 1
            dfs()
            pos[sid] -= 1
            placed.discard(op)
            prefix.pop()

    dfs()
    return results


def _stream_of(data: PlanningInput, op: str) -> str:
    for sid, ops in data.streams.items():
        if op in ops:
            return sid
    raise PlanningError(f"内部错误：算子 {op!r} 不属于任何流")


def _predecessors(
    reach: dict[str, frozenset[str]], op: str
) -> frozenset[str]:
    """可到达 op 的全部算子（op 的 happens-before 前驱）。"""
    preds: set[str] = set()
    for u, succ in reach.items():
        if op in succ:
            preds.add(u)
    return frozenset(preds)


# --------------------------------------------------------------------------- #
# 记录/等待关系：哪些算子要记录事件、每条记录被哪些等待方使用
# --------------------------------------------------------------------------- #
def _record_map(
    waits: list[Wait],
) -> tuple[set[str], dict[str, list[str]]]:
    """返回 (需要记录事件的算子集合, waitee -> 等待它的 waiter 列表)。"""
    recorders: set[str] = set()
    waiters_of: dict[str, list[str]] = {}
    for w in waits:
        recorders.add(w.waitee)
        waiters_of.setdefault(w.waitee, []).append(w.waiter)
    for waitee in waiters_of:
        waiters_of[waitee].sort()
    return recorders, waiters_of


def evaluate_order(
    data: PlanningInput,
    order: list[str],
    waits: list[Wait],
) -> InterleavingEval:
    """评估一份提交顺序的切流次数与最少事件句柄峰值。"""
    recorders, waiters_of = _record_map(waits)
    stream_at = {op: _stream_of(data, op) for op in order}
    position = {op: i for i, op in enumerate(order)}

    # 每条记录的“最后一个等待方批次”位置；该批次之后记录方可回收
    close_at: dict[str, int] = {}
    for waitee, waiters in waiters_of.items():
        close_at[waitee] = max(position[w] for w in waiters)

    switches = sum(
        1 for i in range(1, len(order))
        if stream_at[order[i]] != stream_at[order[i - 1]]
    )

    # 逐前缀统计仍打开的记录数（本批先打开自己的记录，再关闭在本批到期的）
    open_records: set[str] = set()
    peak = 0
    for i, op in enumerate(order):
        if op in recorders:
            open_records.add(op)
        for waitee, close_pos in close_at.items():
            if close_pos == i:
                open_records.discard(waitee)
        peak = max(peak, len(open_records))

    return InterleavingEval(order=list(order), switches=switches, peak_handles=peak)


def best_order(
    data: PlanningInput,
    reach: dict[str, frozenset[str]],
    waits: list[Wait],
    max_switches: int,
    max_handles: int,
) -> InterleavingEval | None:
    """在满足两项上限的提交顺序中择优：切流最少，其次句柄峰值最少，
    再次提交序列字典序最小。没有满足上限的顺序时返回 None。"""
    best: InterleavingEval | None = None
    for order in enumerate_orders(data, reach):
        ev = evaluate_order(data, order, waits)
        if ev.switches > max_switches or ev.peak_handles > max_handles:
            continue
        if (
            best is None
            or (ev.switches, ev.peak_handles, ev.order)
            < (best.switches, best.peak_handles, best.order)
        ):
            best = ev
    return best


def _unconstrained_best(
    data: PlanningInput,
    reach: dict[str, frozenset[str]],
    waits: list[Wait],
) -> InterleavingEval:
    """不考虑上限时的最优顺序（用于无解诊断，报告实际可达的最小用量）。"""
    evals = [evaluate_order(data, o, waits) for o in enumerate_orders(data, reach)]
    return min(evals, key=lambda e: (e.switches, e.peak_handles, e.order))


# --------------------------------------------------------------------------- #
# 由提交顺序生成批次清单（含确定性的句柄分配：贪心区间着色）
# --------------------------------------------------------------------------- #
def build_batches(
    data: PlanningInput,
    ev: InterleavingEval,
    waits: list[Wait],
) -> tuple[list[SubmissionBatch], int]:
    recorders, waiters_of = _record_map(waits)
    position = {op: i for i, op in enumerate(ev.order)}
    stream_at = {op: _stream_of(data, op) for op in ev.order}

    # 每条记录的打开位置与关闭位置（关闭位置 = 最后一个等待方批次）
    open_at = {op: position[op] for op in recorders}
    close_at = {
        waitee: max(position[w] for w in waiters)
        for waitee, waiters in waiters_of.items()
    }

    # 区间着色：按记录位置扫描，回收 close 严格早于当前位置的句柄，
    # 取最小空闲编号；同位置多条记录按算子 id 字典序分配，结果确定。
    # 注意只把句柄放入空闲池，不覆盖 handle_of：等待批次仍需通过 waitee
    # 查到它所绑定的那次记录当时分配到的编号。
    handle_of: dict[str, int] = {}
    free: list[int] = []
    recycled: set[str] = set()
    used_max = -1
    for pos in range(len(ev.order)):
        for waitee, close_pos in close_at.items():
            if close_pos < pos and waitee not in recycled:
                free.append(handle_of[waitee])
                recycled.add(waitee)
        opening = sorted(op for op, p in open_at.items() if p == pos)
        for op in opening:
            if free:
                free.sort()
                handle_of[op] = free.pop(0)
            else:
                used_max += 1
                handle_of[op] = used_max

    # waiter -> 它等待的 waitee 列表（同一算子可能等多个事件）
    incoming: dict[str, list[str]] = {}
    for w in waits:
        incoming.setdefault(w.waiter, []).append(w.waitee)
    for waiter in incoming:
        incoming[waiter].sort()

    batches: list[SubmissionBatch] = []
    for seq_no, op in enumerate(ev.order):
        wait_calls = [
            WaitEventCall(event_handle=handle_of[waitee], waitee=waitee)
            for waitee in incoming.get(op, [])
        ]
        record_call = (
            RecordEventCall(event_handle=handle_of[op]) if op in recorders else None
        )
        batches.append(
            SubmissionBatch(
                seq=seq_no,
                stream_id=stream_at[op],
                operator=op,
                wait_events=wait_calls,
                record_event=record_call,
            )
        )
    return batches, used_max + 1


# --------------------------------------------------------------------------- #
# /submit 编排：显存沿用 /plan 规则，原依赖优先，超预算再枚举单条新增等待
# --------------------------------------------------------------------------- #
def submit_plan(
    data: PlanningInput,
    max_event_handles: int,
    max_host_stream_switches: int,
) -> SubmitOutcome:
    validate_input(data)
    if max_event_handles < 0 or max_host_stream_switches < 0:
        raise PlanningError("事件句柄上限与主机切流上限必须为非负整数")

    base_edges = build_base_edges(data)
    base_result, base_reach = _allocate(data, base_edges)
    base_waits = list(data.waits)

    def make_outcome(
        feasible: bool,
        reason: str,
        alloc,
        waits: list[Wait],
        reach,
        added: Wait | None,
        note: str,
        best_effort: int | None = None,
        diagnostics: dict[str, int] | None = None,
    ) -> SubmitOutcome:
        if feasible:
            ev = best_order(
                data, reach, waits,
                max_host_stream_switches, max_event_handles,
            )
            assert ev is not None  # 调用方已确认可下发
            batches, handles_used = build_batches(data, ev, waits)
            return SubmitOutcome(
                feasible=True,
                reason=REASON_OK,
                total_bytes=alloc.total_bytes,
                budget_bytes=data.budget_bytes,
                offsets=dict(alloc.offsets),
                added_wait=added,
                batches=batches,
                event_handles_used=handles_used,
                host_stream_switches=ev.switches,
                max_event_handles=max_event_handles,
                max_host_stream_switches=max_host_stream_switches,
                note=note,
            )
        return SubmitOutcome(
            feasible=False,
            reason=reason,
            # 无解时 total_bytes 反映该原因下可达的最小占用，
            # 与 best_effort_total_bytes 保持同一口径
            total_bytes=best_effort if best_effort is not None else alloc.total_bytes,
            budget_bytes=data.budget_bytes,
            offsets={},
            added_wait=None,
            batches=[],
            event_handles_used=0,
            host_stream_switches=0,
            max_event_handles=max_event_handles,
            max_host_stream_switches=max_host_stream_switches,
            note=note,
            best_effort_total_bytes=best_effort,
            dispatch_diagnostics=diagnostics or {},
        )

    # 1) 原依赖下能排进显存预算：沿用原依赖，绝不新增等待
    if base_result.feasible:
        ev = best_order(
            data, base_reach, base_waits,
            max_host_stream_switches, max_event_handles,
        )
        if ev is not None:
            return make_outcome(
                True, REASON_OK, base_result, base_waits, base_reach, None,
                "原依赖下显存满足预算，且提交顺序满足句柄与切流上限，未新增跨流等待",
            )
        diag = _unconstrained_best(data, base_reach, base_waits)
        return make_outcome(
            False, REASON_DISPATCH, base_result, base_waits, base_reach, None,
            "显存满足预算，但不存在同时满足事件句柄上限与主机切流上限的提交顺序，"
            "下发受限，无解",
            best_effort=base_result.total_bytes,
            diagnostics={
                "min_host_stream_switches": diag.switches,
                "min_event_handles": diag.peak_handles,
            },
        )

    # 2) 原依赖超预算：枚举全部不成环的单条新增等待，逐个做布局 + 可下发性评估
    candidates = candidate_waits(data, base_edges, base_reach)
    memory_fit: list[tuple[int, str, str, Wait, object, object]] = []
    best_memory_total = base_result.total_bytes
    best_memory_wait: Wait | None = None

    for wait in candidates:
        edges = set(base_edges)
        edges.add((wait.waitee, wait.waiter))
        alloc, reach = _allocate(data, edges)
        if alloc.total_bytes < best_memory_total:
            best_memory_total = alloc.total_bytes
            best_memory_wait = wait
        if alloc.feasible:
            memory_fit.append(
                (alloc.total_bytes, wait.waiter, wait.waitee, wait, alloc, reach)
            )

    # 2a) 没有任何单等待方案能排进预算 -> 显存不足
    if not memory_fit:
        # best_memory_wait 为 None 说明没有任何候选能改善占用，沿用原依赖布局
        if best_memory_wait is None:
            return make_outcome(
                False, REASON_MEMORY, base_result, base_waits, base_reach, None,
                "原依赖及任意单条不成环新增等待下，最小显存占用仍超出预算，显存不足，无解",
                best_effort=best_memory_total,
            )
        edges = set(base_edges)
        edges.add((best_memory_wait.waitee, best_memory_wait.waiter))
        best_alloc, best_reach = _allocate(data, edges)
        return make_outcome(
            False, REASON_MEMORY, best_alloc,
            base_waits + [best_memory_wait], best_reach, None,
            "原依赖及任意单条不成环新增等待下，最小显存占用仍超出预算，显存不足，无解",
            best_effort=best_memory_total,
        )

    # 2b) 在“排得下又能下发”的方案中按 /plan 规则择优：
    #     占用最小，相同取 (waiter, waitee) 字典序最小
    dispatchable: list[tuple[int, str, str, Wait, object, object]] = []
    for total, waiter, waitee, wait, alloc, reach in memory_fit:
        ev = best_order(
            data, reach, base_waits + [wait],
            max_host_stream_switches, max_event_handles,
        )
        if ev is not None:
            dispatchable.append((total, waiter, waitee, wait, alloc, reach))

    if not dispatchable:
        # 显存排得下的方案全部无法在两项上限内下发 -> 下发受限
        diag_total, _, _, diag_wait, diag_alloc, diag_reach = min(
            memory_fit, key=lambda x: (x[0], x[1], x[2])
        )
        diag = _unconstrained_best(data, diag_reach, base_waits + [diag_wait])
        return make_outcome(
            False, REASON_DISPATCH, diag_alloc, base_waits + [diag_wait],
            diag_reach, None,
            "存在满足显存预算的单等待方案，但均无法在事件句柄上限与主机切流上限内"
            "下发，下发受限，无解",
            best_effort=diag_total,
            diagnostics={
                "min_host_stream_switches": diag.switches,
                "min_event_handles": diag.peak_handles,
            },
        )

    total, waiter, waitee, wait, alloc, reach = min(
        dispatchable, key=lambda x: (x[0], x[1], x[2])
    )
    return make_outcome(
        True, REASON_OK, alloc, base_waits + [wait], reach, wait,
        f"原依赖下最小占用 {base_result.total_bytes} 字节超出预算，新增跨流等待 "
        f"{wait.waiter!r} 等待 {wait.waitee!r} 完成后占用降至 {alloc.total_bytes} 字节，"
        "且提交顺序满足句柄与切流上限",
    )
