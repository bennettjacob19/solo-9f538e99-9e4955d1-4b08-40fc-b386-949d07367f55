"""FastAPI 离线显存规划接口。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .planner.models import (
    PlanningError,
    PlanningInput,
    TensorSpec,
    Wait,
)
from .planner.service import plan
from .planner.submission import submit_plan

app = FastAPI(
    title="CUDA 双流显存离线规划接口",
    version="1.0.0",
    description=(
        "给定两条 CUDA 流的算子顺序、跨流等待与张量生命周期，"
        "在显存预算内规划各张量的字节偏移；必要时新增一条不成环的跨流等待。"
    ),
)


class WaitIn(BaseModel):
    waiter: str = Field(..., description="等待方算子 id（该算子需等 waitee 完成）")
    waitee: str = Field(..., description="被等待方算子 id")


class TensorIn(BaseModel):
    name: str = Field(..., description="张量 id")
    producer: str = Field(..., description="生产算子 id")
    consumers: list[str] = Field(..., min_length=1, description="全部消费算子 id")
    size_bytes: int = Field(..., gt=0, description="张量字节数")
    alignment: int = Field(..., gt=0, description="偏移对齐要求，必须为 2 的幂")


class PlanRequest(BaseModel):
    streams: dict[str, list[str]] = Field(
        ...,
        description="两条 CUDA 流，key 为流 id，value 为按提交顺序排列的算子 id",
    )
    waits: list[WaitIn] = Field(default_factory=list, description="已有跨流等待")
    tensors: list[TensorIn] = Field(..., min_length=1, description="待规划张量，最多 8 个")
    budget_bytes: int = Field(..., gt=0, description="显存预算（字节）")


class AddedWaitOut(BaseModel):
    waiter: str
    waitee: str


class PlanResponse(BaseModel):
    feasible: bool
    total_bytes: int
    budget_bytes: int
    offsets: dict[str, int]
    added_wait: AddedWaitOut | None
    note: str
    best_effort_total_bytes: int | None = None
    bins: list[list[str]]


# --------------------------------------------------------------------------- #
# /submit：双流 CUDA 主机提交清单
# --------------------------------------------------------------------------- #
class SubmitRequest(PlanRequest):
    max_event_handles: int = Field(
        ..., ge=0, description="允许同时在途（已记录、等待未全部提交）的事件句柄数上限"
    )
    max_host_stream_switches: int = Field(
        ..., ge=0, description="主机提交相邻批次切换 CUDA 流的次数上限"
    )


class WaitEventOut(BaseModel):
    event_handle: int = Field(..., description="等待的事件句柄编号（与某次 record 绑定）")
    waitee: str = Field(..., description="被等待方算子 id（记录该事件的算子）")


class RecordEventOut(BaseModel):
    event_handle: int = Field(..., description="记录所用的事件句柄编号")


class SubmissionBatchOut(BaseModel):
    seq: int = Field(..., description="主机提交序号，从 0 开始")
    stream_id: str = Field(..., description="该批次下发的流 id")
    operator: str = Field(..., description="本批启动的算子 id")
    # 每批固定顺序：先 wait_event（可空），再启动算子，最后按需 record_event
    wait_events: list[WaitEventOut] = Field(default_factory=list)
    record_event: RecordEventOut | None = None


class SubmitResponse(BaseModel):
    feasible: bool
    reason: str = Field(
        ...,
        description="ok=成功；memory=显存不足（任何单等待方案都超预算）；"
        "dispatch=显存可排下但句柄/切流上限内无可下发顺序",
    )
    total_bytes: int
    budget_bytes: int
    offsets: dict[str, int]
    added_wait: AddedWaitOut | None
    batches: list[SubmissionBatchOut]
    event_handles_used: int = Field(..., description="实际分配的事件句柄总数")
    host_stream_switches: int = Field(..., description="实际主机切流次数（相邻异流批次计一次）")
    max_event_handles: int
    max_host_stream_switches: int
    note: str
    best_effort_total_bytes: int | None = None
    dispatch_diagnostics: dict[str, int] = Field(
        default_factory=dict,
        description="无解时给出不受上限约束的最小切流次数与最小句柄峰值，便于放宽参数",
    )


def _to_internal(req: PlanRequest) -> PlanningInput:
    return PlanningInput(
        streams={sid: list(ops) for sid, ops in req.streams.items()},
        waits=[Wait(w.waiter, w.waitee) for w in req.waits],
        tensors=[
            TensorSpec(
                name=t.name,
                producer=t.producer,
                consumers=tuple(t.consumers),
                size=t.size_bytes,
                alignment=t.alignment,
            )
            for t in req.tensors
        ],
        budget_bytes=req.budget_bytes,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/plan", response_model=PlanResponse)
def post_plan(req: PlanRequest) -> PlanResponse | JSONResponse:
    data = _to_internal(req)
    try:
        outcome = plan(data)
    except PlanningError as exc:
        # 输入不合法（结构问题由 pydantic 拦截，这里是语义校验失败）
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return PlanResponse(
        feasible=outcome.feasible,
        total_bytes=outcome.total_bytes,
        budget_bytes=outcome.budget_bytes,
        offsets=outcome.offsets,
        added_wait=(
            AddedWaitOut(
                waiter=outcome.added_wait.waiter,
                waitee=outcome.added_wait.waitee,
            )
            if outcome.added_wait is not None
            else None
        ),
        note=outcome.note,
        best_effort_total_bytes=outcome.best_effort_total_bytes,
        bins=outcome.bins,
    )


@app.post("/submit", response_model=SubmitResponse)
def post_submit(req: SubmitRequest) -> SubmitResponse | JSONResponse:
    data = _to_internal(req)
    try:
        outcome = submit_plan(
            data,
            max_event_handles=req.max_event_handles,
            max_host_stream_switches=req.max_host_stream_switches,
        )
    except PlanningError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return SubmitResponse(
        feasible=outcome.feasible,
        reason=outcome.reason,
        total_bytes=outcome.total_bytes,
        budget_bytes=outcome.budget_bytes,
        offsets=outcome.offsets,
        added_wait=(
            AddedWaitOut(
                waiter=outcome.added_wait.waiter,
                waitee=outcome.added_wait.waitee,
            )
            if outcome.added_wait is not None
            else None
        ),
        batches=[
            SubmissionBatchOut(
                seq=b.seq,
                stream_id=b.stream_id,
                operator=b.operator,
                wait_events=[
                    WaitEventOut(event_handle=w.event_handle, waitee=w.waitee)
                    for w in b.wait_events
                ],
                record_event=(
                    RecordEventOut(event_handle=b.record_event.event_handle)
                    if b.record_event is not None
                    else None
                ),
            )
            for b in outcome.batches
        ],
        event_handles_used=outcome.event_handles_used,
        host_stream_switches=outcome.host_stream_switches,
        max_event_handles=outcome.max_event_handles,
        max_host_stream_switches=outcome.max_host_stream_switches,
        note=outcome.note,
        best_effort_total_bytes=outcome.best_effort_total_bytes,
        dispatch_diagnostics=outcome.dispatch_diagnostics,
    )
