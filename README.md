# CUDA 双流显存离线规划服务

在边缘推理部署前，对两条 CUDA 流的算子/张量做**离线显存复用规划**，并进一步生成
**主机侧 CUDA 提交清单**。服务提供两个端点，均无需 GPU 即可试用：

- `POST /plan`：给定两条流的算子顺序、已有的跨流等待（依赖）、每个张量的生产算子
  与全部消费算子、字节数、偏移对齐要求与显存预算，返回每个张量在同一块显存中的
  字节偏移；当原依赖下任何合法时序都无法满足预算时，允许新增**一条不成环**的跨流
  等待后再规划，仍然排不下则报告无解。
- `POST /submit`：沿用 `/plan` 的全部输入与显存布局算法，额外接收**事件句柄数**与
  **主机切流次数**两项上限，返回按主机提交顺序排列的算子批次清单、张量偏移与两项
  资源的实际用量；无解时区分“显存不足”与“下发受限”。

## 规划语义（核心规则）

1. 每个张量从其**生产算子开始**到**最后一个消费算子执行完成为止**都算在用（live）。
2. 在**任何符合依赖关系的时序**下，同时在用的两个张量所占字节区间不得重叠。
   实现上用偏序（happens-before）判定：张量 `a` 先于 `b` 死亡、`b` 才出生，即
   `last_consumer(a)` 可到达 `producer(b)`，二者才允许复用同一段显存；方向反过来同理。
   两个方向都不可达时，存在让二者同时在用的合法时序，必须错开。
3. 流内按提交顺序建立全序依赖；已有跨流等待是额外的有向边。整条依赖图必须是 DAG。
4. 仅当原依赖下最优布局仍超预算时，才允许增加一条跨流等待边；只考虑不成环的边。
   在所有可行方案中取占用最小的，占用相同取（等待方算子 id, 被等方算子 id）字典序最小的。
5. 每个张量的偏移必须满足自身对齐要求（对齐量必须为 2 的幂）。
6. 约束：算子总数 ≤ 12，张量总数 ≤ 8，两条流各自的算子 id 不允许重复。

## 目录

```
app/
  main.py          FastAPI 入口与请求模型（/plan 与 /submit）
  planner/
    models.py      输入校验后的内部数据结构（含提交批次/事件调用）
    solver.py      依赖图、干扰图与最优显存布局（标准库实现）
    service.py     /plan 编排：原依赖优先，必要时加一条等待
    submission.py  /submit 编排：提交顺序枚举、句柄/切流评估与清单生成
tests/             pytest 自测（含属性化随机校验）
Dockerfile
docker-compose.yml
requirements.txt
```

## 本地运行

```bash
docker compose up --build
# 打开 http://localhost:8000/docs 提交规划请求
```

不用 Docker 时：

```bash
python3 -m venv --without-pip .venv && .venv/bin/python <(curl -sS https://bootstrap.pypa.io/get-pip.py)
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 自测

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

## 请求示例

### `POST /plan`

请求体见 `examples/request.json`，响应字段：

- `feasible`: 是否在预算内排下
- `total_bytes`: 本次布局实际占用的连续字节数
- `budget_bytes`: 回显预算
- `offsets`: 张量 id -> 字节偏移
- `added_wait`: 新增的跨流等待，未新增为 `null`
- `note`: 说明信息
- 无解时 `feasible=false`，`offsets` 为空，并通过 `best_effort_total_bytes`
  给出（加至多一条等待后）能达到的最小占用，便于调用方判断预算缺口。

### `POST /submit`：双流推理的 CUDA 提交清单

请求体在 `/plan` 全部字段之上增加两项上限（示例见
`examples/submit_request.json`）：

- `max_event_handles`：允许同时在途（已记录、等待尚未全部提交）的事件句柄数上限
- `max_host_stream_switches`：主机提交相邻批次切换 CUDA 流的次数上限

#### 提交语义（核心规则）

1. 每个算子对应一个**批次**，批次带流 id，整体按**主机提交顺序**排列；同流批次的
   相对顺序固定为流内提交顺序，两流之间可交错。合法提交顺序即依赖 DAG
   （流内顺序边 + 跨流等待边）的拓扑序。
2. 每个批次在同一流里按固定三段提交：**先** `cudaStreamWaitEvent`（本算子的跨流
   等待，可空）**，再**启动算子，**随后**按需 `cudaEventRecord`（本算子被其他流
   等待时记录一次）。
3. 已有和新增的跨流等待都必须绑定**所等算子已提交的那次记录**：等待方批次中
   `wait_event` 的句柄编号等于被等方批次 `record_event` 的句柄编号；拓扑序保证
   被等方批次先提交，因此主机不会等待一个尚未记录的事件。复用句柄时同样绑定到
   对应那次记录，句柄编号在记录与等待之间一致。
4. 提交过程中主机**不等 GPU 完成**（无任何同步调用）；设备侧依赖只来自流内顺序
   与方案中的等待。
5. **句柄复用**：一条记录在“最后一个等待它的批次提交之后”即可回收句柄。对固定
   提交顺序，最少句柄数等于逐前缀“已记录且等待未全部提交”记录数的峰值，实现用
   贪心区间着色给出确定性的句柄编号（编号从 0 起连续）。
6. **切流计数**：相邻两个批次属于不同流即计一次。
7. 提交顺序在满足两项上限的方案中择优：切流次数最少，其次句柄峰值最少，再次提交
   序列字典序最小，结果确定。
8. 显存布局完全沿用 `/plan` 的算法：原依赖下能排进预算就沿用原依赖、绝不新增
   等待；超预算时，在**既排得下（满足预算）又能下发（满足两项上限）**的单条新增
   不成环等待方案中，按 `/plan` 规则择优（占用最小，相同取
   `(waiter, waitee)` 字典序最小）。

#### 响应字段

- `feasible` / `reason`：成功为 `ok`；无解时区分
  - `memory`：显存不足——原依赖及任意单条新增等待下最小占用仍超预算
  - `dispatch`：下发受限——存在满足显存预算的方案，但没有任何提交顺序能同时
    满足句柄与切流上限
- `batches`：提交清单，元素含 `seq`（从 0 起）、`stream_id`、`operator`、
  `wait_events`（`event_handle` + `waitee`）、`record_event`（句柄编号或 `null`）
- `offsets`：张量 id -> 字节偏移（与 `/plan` 同一布局算法的结果）
- `event_handles_used` / `host_stream_switches`：两项资源的**实际用量**
- `added_wait`：实际采用的新增等待，未新增为 `null`
- 无解时不返回部分清单（`batches` 与 `offsets` 均为空）：
  - `reason=memory` 时 `best_effort_total_bytes` 给出可达最小占用
  - `reason=dispatch` 时 `dispatch_diagnostics` 给出不受上限约束的
    `min_event_handles` 与 `min_host_stream_switches`，便于调用方判断该放宽哪一项

#### 试用

无需 GPU，本地起服务后在 http://localhost:8000/docs 找到 `POST /submit` 即可
在线试填；也可用示例请求：

```bash
curl -s localhost:8000/submit -H 'content-type: application/json' \
  --data @examples/submit_request.json
```
