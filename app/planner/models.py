"""显存复用规划的核心数据结构（只依赖标准库）。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Wait:
    """跨流等待：waiter 算子必须等 waitee 算子完成后才能开始。"""

    waiter: str
    waitee: str


@dataclass(frozen=True)
class TensorSpec:
    name: str
    producer: str
    consumers: tuple[str, ...]
    size: int
    alignment: int


@dataclass
class PlanningInput:
    # 流 id -> 该流按提交顺序排列的算子 id
    streams: dict[str, list[str]]
    waits: list[Wait]
    tensors: list[TensorSpec]
    budget_bytes: int

    def all_operators(self) -> list[str]:
        ops: list[str] = []
        for stream_id in sorted(self.streams):
            ops.extend(self.streams[stream_id])
        return ops


@dataclass
class AllocationResult:
    """一次确定依赖图下的最优布局结果。"""

    feasible: bool
    total_bytes: int
    offsets: dict[str, int]
    # 调试/诊断用：每个“生命周期互不相交组”包含的张量
    bins: list[list[str]] = field(default_factory=list)


@dataclass
class PlanOutcome:
    feasible: bool
    total_bytes: int
    budget_bytes: int
    offsets: dict[str, int]
    added_wait: Wait | None
    note: str
    best_effort_total_bytes: int | None = None
    bins: list[list[str]] = field(default_factory=list)


class PlanningError(ValueError):
    """输入本身不合法（与“排不下”的无解不同，这里对应 HTTP 422）。"""


# --------------------------------------------------------------------------- #
# 双流 CUDA 提交清单（/submit）的数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WaitEventCall:
    """一次 cudaStreamWaitEvent：本批次算子在启动前等待某个已记录事件。"""

    event_handle: int
    waitee: str  # 被等待的算子（记录该事件的算子）


@dataclass(frozen=True)
class RecordEventCall:
    """一次 cudaEventRecord：算子启动后在同流上记录事件，供后续跨流等待绑定。"""

    event_handle: int


@dataclass
class SubmissionBatch:
    """主机按提交顺序下发的一个批次（同一流上的一个算子）。

    每批内部顺序固定：先提交全部 wait_event（跨流等待），再启动算子，
    最后按需 record_event。
    """

    seq: int
    stream_id: str
    operator: str
    wait_events: list[WaitEventCall] = field(default_factory=list)
    record_event: RecordEventCall | None = None


@dataclass
class InterleavingEval:
    """一份主机提交顺序（两流算子的交错序列）的资源用量评估。"""

    order: list[str]
    switches: int          # 相邻异流批次的切流次数
    peak_handles: int      # 任一时刻“已记录、等待尚未全部提交”的最大事件数


@dataclass
class SubmitOutcome:
    """/submit 的编排结果。"""

    feasible: bool
    reason: str  # "ok" | "memory" | "dispatch"
    total_bytes: int
    budget_bytes: int
    offsets: dict[str, int]
    added_wait: Wait | None
    batches: list[SubmissionBatch]
    event_handles_used: int
    host_stream_switches: int
    max_event_handles: int
    max_host_stream_switches: int
    note: str
    # 无解诊断
    best_effort_total_bytes: int | None = None
    dispatch_diagnostics: dict[str, "int"] = field(default_factory=dict)
