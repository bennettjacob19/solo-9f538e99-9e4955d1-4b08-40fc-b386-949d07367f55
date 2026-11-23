"""/submit 的 FastAPI 接口层测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _payload(**overrides) -> dict:
    base = {
        "streams": {
            "stream_a": ["a1", "a2", "a3"],
            "stream_b": ["b1", "b2", "b3"],
        },
        "waits": [{"waiter": "b2", "waitee": "a1"}],
        "tensors": [
            {
                "name": "TA", "producer": "a1", "consumers": ["a2"],
                "size_bytes": 100, "alignment": 1,
            },
            {
                "name": "TB", "producer": "b1", "consumers": ["b2"],
                "size_bytes": 100, "alignment": 1,
            },
        ],
        "budget_bytes": 200,
        "max_event_handles": 4,
        "max_host_stream_switches": 4,
    }
    base.update(overrides)
    return base


def test_submit_ok_batch_shape_and_binding():
    r = client.post("/submit", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is True
    assert body["reason"] == "ok"
    assert body["added_wait"] is None
    assert set(body["offsets"]) == {"TA", "TB"}

    batches = body["batches"]
    # 序号连续、覆盖全部 6 个算子、带流 id
    assert [b["seq"] for b in batches] == list(range(6))
    assert sorted(b["operator"] for b in batches) == [
        "a1", "a2", "a3", "b1", "b2", "b3"
    ]
    for b in batches:
        assert b["stream_id"] in ("stream_a", "stream_b")

    # b2 等 a1：a1 批次有 record，b2 批次的 wait 绑定同一句柄，且 a1 先提交
    a1 = next(b for b in batches if b["operator"] == "a1")
    b2 = next(b for b in batches if b["operator"] == "b2")
    assert a1["record_event"] is not None
    handle = a1["record_event"]["event_handle"]
    assert b2["wait_events"] == [{"event_handle": handle, "waitee": "a1"}]
    assert a1["seq"] < b2["seq"]

    # 两项实际用量回显且不超上限
    assert body["event_handles_used"] <= 4
    assert body["host_stream_switches"] <= 4
    assert body["event_handles_used"] >= 1


def test_submit_dispatch_infeasible_no_partial_checklist():
    # 切流上限 0：含跨流等待的任何顺序都至少切流一次
    r = client.post("/submit", json=_payload(max_host_stream_switches=0))
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is False
    assert body["reason"] == "dispatch"
    assert body["batches"] == []
    assert body["offsets"] == {}
    assert body["dispatch_diagnostics"]["min_host_stream_switches"] >= 1


def test_submit_memory_infeasible():
    r = client.post(
        "/submit",
        json=_payload(
            budget_bytes=50,
            tensors=[
                {
                    "name": "TA", "producer": "a1", "consumers": ["a2"],
                    "size_bytes": 300, "alignment": 1,
                },
                {
                    "name": "TB", "producer": "b1", "consumers": ["b2"],
                    "size_bytes": 300, "alignment": 1,
                },
            ],
        ),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is False
    assert body["reason"] == "memory"
    assert body["best_effort_total_bytes"] == 300
    assert body["batches"] == []


def test_submit_adds_wait_under_tight_budget():
    # 不带已有等待，与 /plan 的字典序择优场景完全对齐
    r = client.post("/submit", json=_payload(budget_bytes=100, waits=[]))
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is True
    # 与 /plan 完全一致的字典序择优
    assert body["added_wait"] == {"waiter": "a1", "waitee": "b2"}
    assert body["total_bytes"] == 100
    # 新增等待同样要落到清单里：b2 批次记录事件，a1 批次等待该事件
    batches = body["batches"]
    a1 = next(b for b in batches if b["operator"] == "a1")
    b2 = next(b for b in batches if b["operator"] == "b2")
    assert b2["record_event"] is not None
    assert any(w["waitee"] == "b2" for w in a1["wait_events"])
    # 等待绑定的句柄必须来自 b2 那次记录
    handle = b2["record_event"]["event_handle"]
    assert {"event_handle": handle, "waitee": "b2"} in a1["wait_events"]


def test_submit_pydantic_422_on_negative_cap():
    r = client.post("/submit", json=_payload(max_event_handles=-1))
    assert r.status_code == 422


def test_submit_semantic_422():
    payload = _payload(
        tensors=[
            {"name": "T", "producer": "b2", "consumers": ["a1"], "size_bytes": 10, "alignment": 1}
        ]
    )
    r = client.post("/submit", json=payload)
    assert r.status_code == 422
    assert "数据竞争" in r.json()["detail"]


def test_submit_documented_in_openapi():
    spec = client.get("/openapi.json").json()
    assert "/submit" in spec["paths"]
    assert "/plan" in spec["paths"]
    # /docs 页面可打开（任务要求在 /docs 提供该能力）
    assert client.get("/docs").status_code == 200
