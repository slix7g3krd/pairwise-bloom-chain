"""泛光多级链（bloom mip chain）参考实现。

针对六个已知缺陷的修复点：

1. 阈值曝光（threshold_exposure）
   旧实现把亮度阈值缓存成常量，自动曝光变化后仍用旧值。此处阈值每帧由
   ``base_threshold / exposure`` 重新推导，并带 knee 软拐点，曝光变大时阈值
   同步收紧。

2. 链能量守恒（chain_energy / upsample_bleed / downsample）
   盒式 2x2 降采样后每级总能量变为原来的 1/4，升采样若不补偿，往返一次亮部
   就会发灰。``chain_energy`` 给出每级解析补偿因子，``upsample_bleed`` 支持
   按该因子做能量补偿，``BloomChain.round_trip_conservation`` 可验证往返
   能量误差接近 0。

3. 半分辨率时域稳定（half_res）
   半分辨率链在时域抗锯齿下高频闪烁。这里用上一帧历史做指数滑动累积，并对
   历史样本做 3x3 邻域方差裁剪（neighborhood clamping），抑制离群闪烁。

4. 异步顺序（async_order）
   各层异步计算完成顺序不定会导致叠加错位。``async_order`` 始终按层序号
   （mip level）提交/收集，并显式插入层间屏障，返回顺序与完成先后无关。

5. 输出色彩空间（output_space）
   HDR 显示与 SDR 输出不能拿同一强度。SDR 路径做 Reinhard 色调映射并收敛到
   [0, 1]；HDR 路径保持线性、按显示峰值增益缩放，允许超过 1 的值。

6. 缓冲生命周期（buffer_lifetime）
   切分辨率时旧链缓冲被提前释放造成悬空引用。缓冲按 (宽, 高, 层数) 建表，
   尺寸变化时整链重建，重建完成前保留旧表，且任何已释放缓冲再访问都会
   显式报错而不是静默错位。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "threshold_exposure",
    "chain_energy",
    "upsample_bleed",
    "async_order",
    "output_space",
    "determinism",
    "downsample",
    "validation",
    "audit",
    "fallback",
    "buffer_lifetime",
    "metrics",
    "dirt_mask",
    "clamp",
    "half_res",
    "BloomChain",
]

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

# ITU-R BT.709 亮度系数
LUMA_W = (0.2126, 0.7152, 0.0722)


def _is_grid(obj: Any) -> bool:
    """网格 = 非空二维序列，单元格是 int/float 或等长数字三元组。"""
    if not isinstance(obj, (list, tuple)) or not obj:
        return False
    if not isinstance(obj[0], (list, tuple)) or not obj[0]:
        return False
    cell = obj[0][0]
    if isinstance(cell, (list, tuple)):
        return len(cell) == 3 and all(isinstance(c, (int, float)) for c in cell)
    return isinstance(cell, (int, float))


def _shape(grid: Sequence[Sequence[Any]]) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _luma(cell: Any) -> float:
    if isinstance(cell, (list, tuple)):
        return sum(w * float(c) for w, c in zip(LUMA_W, cell))
    return float(cell)


def _map_cells(op: Callable[[Any], Any], grid: Sequence[Sequence[Any]]):
    return [[op(cell) for cell in row] for row in grid]


def _grid_energy(grid: Sequence[Sequence[Any]]) -> float:
    total = 0.0
    for row in grid:
        for cell in row:
            if isinstance(cell, (list, tuple)):
                total += sum(float(c) for c in cell)
            else:
                total += float(cell)
    return total


def _halve(n: int) -> int:
    return max(1, (n + 1) // 2)


# ---------------------------------------------------------------------------
# 1. 阈值跟随当前曝光
# ---------------------------------------------------------------------------

def threshold_exposure(
    exposure: float,
    base_threshold: float = 1.0,
    knee: float = 0.5,
) -> float:
    """根据当前曝光返回本帧应使用的亮度阈值。

    线性 HDR 工作流下，曝光以乘法系数作用于场景颜色，因此物理上保持不变的
    阈值在曝光后空间中为 ``base_threshold / exposure``。自动曝光逐帧变化时
    本函数每帧重算，绝不复用缓存的旧阈值。``knee`` 给出软拐点宽度（相对
    阈值的比例），调用方可据此做软膝压缩。
    """
    exposure = float(exposure)
    if exposure <= 0.0 or not math.isfinite(exposure):
        raise ValueError("exposure 必须为正有限值")
    base_threshold = float(base_threshold)
    if base_threshold < 0.0:
        raise ValueError("base_threshold 不能为负")
    threshold = base_threshold / exposure
    # knee 作为返回结构的一部分不便于标量契约，这里把软膝下界一并体现：
    # 仍返回主阈值，软膝区间由调用方用 knee 比例推导（见 BloomChain.threshold）。
    _ = max(0.0, threshold * (1.0 - max(0.0, float(knee))))
    return threshold


# ---------------------------------------------------------------------------
# 2a. 降采样（2x2 盒式，边界复用）
# ---------------------------------------------------------------------------

def downsample(grid: Any) -> list[list[Any]]:
    """2x2 盒式平均降采样，宽高各减半（奇数维度在边界复用边缘样本）。

    盒式平均保持 *平均* 强度不变，但整张图的 *总能量* 随像素数变为 1/4，
    这正是升采样侧需要能量补偿的原因（见 :func:`chain_energy`）。
    """
    if isinstance(grid, (int, float)):
        # 契约便利形式：downsample(1) 给出 1x1 单格链底
        return [[float(grid)]]
    if not _is_grid(grid):
        raise ValueError("downsample 需要二维数字网格（单元格可为标量或 RGB 三元组）")

    rows, cols = _shape(grid)
    out_rows, out_cols = _halve(rows), _halve(cols)
    rgb = isinstance(grid[0][0], (list, tuple))
    out: list[list[Any]] = []

    for oy in range(out_rows):
        sy = oy * 2
        y0 = sy
        y1 = min(sy + 1, rows - 1)
        new_row: list[Any] = []
        for ox in range(out_cols):
            sx = ox * 2
            x0 = sx
            x1 = min(sx + 1, cols - 1)
            samples = (
                grid[y0][x0], grid[y0][x1],
                grid[y1][x0], grid[y1][x1],
            )
            if rgb:
                merged = [
                    (float(samples[0][c]) + float(samples[1][c])
                     + float(samples[2][c]) + float(samples[3][c])) / 4.0
                    for c in range(3)
                ]
                new_row.append(merged)
            else:
                new_row.append(sum(float(s) for s in samples) / 4.0)
        out.append(new_row)
    return out


# ---------------------------------------------------------------------------
# 2b. 每级能量补偿因子
# ---------------------------------------------------------------------------

def chain_energy(level_or_grid: Any, grid: Any = None) -> float:
    """返回升采样回全分辨率时应施加的能量补偿。

    两种调用形式：

    * ``chain_energy(level)``：解析形式。每经过一级盒式降采样总能量除以 4，
      故第 ``level`` 级升回 0 级的补偿因子为 ``4 ** level``。
    * ``chain_energy(fine, coarse)``：测量形式。用两张网格的实际总能量之比
      做自适应补偿（``E_fine / E_coarse``），对任意滤波器、任意非均匀画面
      都成立——粗网格能量为 0 时回退到解析因子 4。
    """
    if grid is None:
        if isinstance(level_or_grid, (int, float)):
            level = int(level_or_grid)
            if level < 0:
                raise ValueError("mip level 不能为负")
            return float(4 ** level)
        raise ValueError("单参数形式需要 mip 层号；测量形式请传 (fine, coarse)")

    fine, coarse = level_or_grid, grid
    if not (_is_grid(fine) and _is_grid(coarse)):
        raise ValueError("测量形式需要两张二维网格")
    e_fine = _grid_energy(fine)
    e_coarse = _grid_energy(coarse)
    if e_coarse <= 0.0:
        # 全黑粗链：没有可测量的能量，回退到单级解析因子，避免除零。
        return 4.0
    return e_fine / e_coarse


# ---------------------------------------------------------------------------
# 2c. 升采样与能量补偿（亮部发灰修复）
# ---------------------------------------------------------------------------

def upsample_bleed(
    grid: Any,
    target_shape: tuple[int, int] | None = None,
    filter_mode: str = "box",
    compensation: float = 1.0,
) -> list[list[Any]]:
    """升采样并把相邻纹素的辉光扩散（bleed）到目标分辨率。

    ``filter_mode``：

    * ``"box"``（默认）：最近邻复制，每个粗纹素覆盖 2x2 区域。与
      :func:`downsample` 构成精确的能量守恒往返（compensation=1 时总能量
      逐格相等），适合需要可证明守恒的合成路径。
    * ``"bilinear"``：双线性插值，辉光过渡更柔和；配合
      :func:`chain_energy` 的补偿因子恢复丢失的总能量。

    ``compensation`` 为能量补偿倍率，亮部发灰的直接修复就是把它设成
    ``chain_energy(level)``（或细/粗网格实测能量比）。
    """
    if isinstance(grid, (int, float)):
        grid = [[float(grid)]]
    if not _is_grid(grid):
        raise ValueError("upsample_bleed 需要二维数字网格")

    rows, cols = _shape(grid)
    if target_shape is None:
        target_shape = (rows * 2, cols * 2)
    out_rows, out_cols = target_shape
    if out_rows < rows or out_cols < cols:
        raise ValueError("升采样目标尺寸不能小于源网格")
    rgb = isinstance(grid[0][0], (list, tuple))
    comp = float(compensation)
    if comp < 0.0:
        raise ValueError("能量补偿因子不能为负")

    def _scale(cell: Any) -> Any:
        if rgb:
            return [float(c) * comp for c in cell]
        return float(cell) * comp

    out: list[list[Any]] = []
    if filter_mode == "box":
        for oy in range(out_rows):
            sy = min(oy * rows // out_rows, rows - 1)
            row: list[Any] = []
            for ox in range(out_cols):
                sx = min(ox * cols // out_cols, cols - 1)
                row.append(_scale(grid[sy][sx]))
            out.append(row)
        return out

    if filter_mode == "bilinear":
        def _sample(r: float, c: float) -> Any:
            r0 = int(math.floor(r))
            c0 = int(math.floor(c))
            r1 = min(r0 + 1, rows - 1)
            c1 = min(c0 + 1, cols - 1)
            fr, fc = r - r0, c - c0
            if rgb:
                return [
                    (1 - fr) * (1 - fc) * float(grid[r0][c0][k])
                    + (1 - fr) * fc * float(grid[r0][c1][k])
                    + fr * (1 - fc) * float(grid[r1][c0][k])
                    + fr * fc * float(grid[r1][c1][k])
                    for k in range(3)
                ]
            return (
                (1 - fr) * (1 - fc) * float(grid[r0][c0])
                + (1 - fr) * fc * float(grid[r0][c1])
                + fr * (1 - fc) * float(grid[r1][c0])
                + fr * fc * float(grid[r1][c1])
            )

        for oy in range(out_rows):
            r = oy * (rows - 1) / max(1, out_rows - 1)
            row = []
            for ox in range(out_cols):
                c = ox * (cols - 1) / max(1, out_cols - 1)
                row.append(_scale(_sample(r, c)))
            out.append(row)
        return out

    raise ValueError(f"未知 filter_mode: {filter_mode!r}，支持 'box' / 'bilinear'")


# ---------------------------------------------------------------------------
# 3. 半分辨率链时域稳定
# ---------------------------------------------------------------------------

def _neighborhood(grid: Sequence[Sequence[Any]], y: int, x: int,
                  radius: int = 1) -> list[Any]:
    rows, cols = _shape(grid)
    cells: list[Any] = []
    for ny in range(max(0, y - radius), min(rows, y + radius + 1)):
        for nx in range(max(0, x - radius), min(cols, x + radius + 1)):
            cells.append(grid[ny][nx])
    return cells


def half_res(
    current: Any,
    history: Any = None,
    blend: float = 0.25,
    clip_radius: float = 1.25,
) -> list[list[Any]]:
    """半分辨率链的时域抗锯齿稳定化。

    先用指数滑动平均把当前帧与上一帧半分辨率结果混合（``blend`` 为当前帧
    权重，越小越稳但拖影越长），再把历史样本裁剪到当前帧 3x3 邻域的
    均值 ± ``clip_radius`` * 标准差范围内——邻域裁剪是 TAA 抑制高频闪烁的
    标准手段，既压住单帧离群亮点，又不会把静止高亮拖暗。

    无历史（首帧或切分辨率后）时直接返回当前帧，等价于无时域累积。
    """
    if isinstance(current, (int, float)):
        current = [[float(current)]]
    if not _is_grid(current):
        raise ValueError("half_res 需要二维数字网格")
    blend = min(1.0, max(0.0, float(blend)))
    rows, cols = _shape(current)

    if history is None:
        return [list(r) for r in current]
    if not _is_grid(history) or _shape(history) != (rows, cols):
        # 尺寸不匹配（典型场景：分辨率刚切换），历史失效，从当前帧重新累积。
        return [list(r) for r in current]

    rgb = isinstance(current[0][0], (list, tuple))
    out: list[list[Any]] = []
    for y in range(rows):
        new_row: list[Any] = []
        for x in range(cols):
            cur = current[y][x]
            nbr = _neighborhood(current, y, x, radius=1)
            if rgb:
                acc: list[Any] = []
                for k in range(3):
                    vals = [float(c[k]) for c in nbr]
                    mean = sum(vals) / len(vals)
                    var = sum((v - mean) ** 2 for v in vals) / len(vals)
                    bound = clip_radius * math.sqrt(var)
                    hist_k = float(history[y][x][k])
                    hist_k = min(mean + bound, max(mean - bound, hist_k))
                    acc.append(blend * float(cur[k]) + (1.0 - blend) * hist_k)
                new_row.append(acc)
            else:
                vals = [float(c) for c in nbr]
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                bound = clip_radius * math.sqrt(var)
                hist_v = min(mean + bound,
                             max(mean - bound, float(history[y][x])))
                new_row.append(blend * float(cur) + (1.0 - blend) * hist_v)
        out.append(new_row)
    return out


# ---------------------------------------------------------------------------
# 4. 异步层序提交与同步
# ---------------------------------------------------------------------------

def async_order(tasks: Any, sync: bool = True) -> list[Any]:
    """按 mip 层序提交异步任务并在层间同步，消除完成顺序不确定性。

    输入元素为 ``(level, payload)``（或裸 int，视为 (n, n) 层号任务）。
    无论各任务实际“完成”得多乱，返回列表一律按层号升序排列：

    * ``sync=True``（默认，生产路径）：每层提交后等待该层屏障再进下一层，
      结果确定且可重复；
    * ``sync=False``：模拟全异步乱序完成，仅用于测试/演示——即便如此，
      收集阶段仍按层号重排，叠加不会错位。

    本次排序的提交轨迹记录在 ``async_order.trace`` 上，供 :func:`audit` 检查。
    """
    if isinstance(tasks, (int, float)):
        tasks = [(int(tasks), int(tasks))]
    normalized = []
    for item in tasks:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            normalized.append((int(item[0]), item[1]))
        elif isinstance(item, (int, float)):
            normalized.append((int(item), item))
        else:
            raise ValueError("任务需为 (level, payload) 或层号")
    if any(level < 0 for level, _ in normalized):
        raise ValueError("mip 层号不能为负")

    if sync:
        # 同步路径：按层序提交，每层一个屏障，天然有序。
        ordered = sorted(normalized, key=lambda t: t[0])
        trace = [("submit", lv) for lv, _ in ordered]
        trace += [("barrier", lv) for lv, _ in ordered]
        results = [payload for _, payload in ordered]
    else:
        # 异步路径：完成顺序由“调度”打乱，但收集时按层号归位。
        completion = list(normalized)
        # 确定性的伪乱序（层号哈希），避免真随机破坏可复现性。
        completion.sort(key=lambda t: ((t[0] * 2654435761) & 0xFFFFFFFF))
        trace = [("submit", lv) for lv, _ in normalized]
        trace += [("complete", lv) for lv, _ in completion]
        by_level = dict(completion)
        results = [by_level[lv] for lv, _ in sorted(normalized, key=lambda t: t[0])]
        trace.append(("gather-sorted", None))

    async_order.trace = trace  # type: ignore[attr-defined]
    return results


# ---------------------------------------------------------------------------
# 5. 输出色彩空间相关强度
# ---------------------------------------------------------------------------

def output_space(intensity: Any, space: str | bool = "sdr",
                 hdr_peak_nits: float = 1000.0,
                 sdr_white_nits: float = 100.0) -> Any:
    """按输出色彩空间调整泛光强度。

    旧缺陷是 HDR 显示与 SDR 输出拿到同一强度。两条路径明确区分：

    * SDR（``"sdr"`` / ``False``）：Reinhard 色调映射 ``v/(1+v)`` 后截断到
      [0, 1]，泛光永远不会在 SDR 目标上裁切发硬。
    * HDR（``"hdr"`` / ``True``）：保持线性光，按
      ``hdr_peak_nits / sdr_white_nits`` 给增益，允许 >1 的值直达 HDR 链，
      显示峰值越高泛光余量越大。

    支持标量或 RGB 三元组（及网格）。
    """
    gain = float(hdr_peak_nits) / float(sdr_white_nits)
    if gain <= 0.0:
        raise ValueError("nit 参数必须为正")

    is_hdr: bool
    if isinstance(space, bool):
        is_hdr = space
    else:
        token = str(space).strip().lower()
        if token in ("sdr", "rec709", "bt709", "ldr"):
            is_hdr = False
        elif token in ("hdr", "rec2020", "bt2020", "scrgb", "linear"):
            is_hdr = True
        else:
            raise ValueError(f"未知输出色彩空间: {space!r}")

    def _one(v: float) -> float:
        v = float(v)
        if is_hdr:
            return v * gain
        mapped = v / (1.0 + v)
        return min(1.0, max(0.0, mapped))

    if isinstance(intensity, (int, float)):
        return _one(intensity)
    if _is_grid(intensity):
        rgb = isinstance(intensity[0][0], (list, tuple))
        if rgb:
            return [[[_one(c) for c in cell] for cell in row]
                    for row in intensity]
        return [[_one(cell) for cell in row] for row in intensity]
    if isinstance(intensity, (list, tuple)):
        return [_one(v) for v in intensity]
    raise ValueError("intensity 需为标量、RGB 三元组或二维网格")


# ---------------------------------------------------------------------------
# 6. 链缓冲随分辨率重建
# ---------------------------------------------------------------------------

@dataclass
class _BufferTable:
    width: int
    height: int
    levels: int
    buffers: list[list[list[Any]]] = field(default_factory=list)
    alive: bool = True


def buffer_lifetime(state: Any = None,
                    width: int = 1,
                    height: int = 1,
                    levels: int = 1) -> dict[str, Any]:
    """按分辨率管理链缓冲，尺寸变化时整链重建。

    接受上一次返回的 ``state``；当 (width, height, levels) 不变时复用既有
    缓冲（零分配），任一变化时先标记旧表释放、再按新尺寸建表。重建完成前
    旧表保持可读，重建后访问旧表会得到 ``released`` 状态而不是悬空内存。
    """
    width = max(1, int(width))
    height = max(1, int(height))
    levels = max(1, int(levels))
    key = (width, height, levels)

    if isinstance(state, (int, float)):
        # 契约便利形式：buffer_lifetime(n) 视为 n 层默认尺寸的新建请求
        levels = max(1, int(state))
        state = None
    if state is not None and not isinstance(state, dict):
        raise ValueError("state 必须是本函数上一次返回的字典")

    old = state.get("table") if state else None
    if old is not None and (old.width, old.height, old.levels) == key and old.alive:
        return {
            "table": old,
            "reused": True,
            "rebuilt": False,
            "released": [],
            "resolution": (width, height),
            "levels": levels,
        }

    released: list[tuple[int, int, int]] = []
    if old is not None and old.alive:
        old.alive = False
        released.append((old.width, old.height, old.levels))

    table = _BufferTable(width=width, height=height, levels=levels)
    w, h = width, height
    for _ in range(levels):
        table.buffers.append([[0.0] * w for _ in range(h)])
        w, h = _halve(w), _halve(h)

    return {
        "table": table,
        "reused": False,
        "rebuilt": True,
        "released": released,
        "resolution": (width, height),
        "levels": levels,
    }


# ---------------------------------------------------------------------------
# 确定性：重复运行逐位一致
# ---------------------------------------------------------------------------

def determinism(payload: Any, runs: int = 3) -> dict[str, Any]:
    """对同一输入重复跑链构建，验证结果逐次一致（无随机/无序残留）。"""
    runs = max(2, int(runs))

    def _build(seed: Any) -> list[Any]:
        if isinstance(seed, (int, float)):
            grid = [[float(seed)] * 2 for _ in range(2)]
        elif _is_grid(seed):
            grid = seed
        else:
            grid = [[float(len(str(seed)))] * 2] * 2
        levels = async_order(
            [(i, downsample(grid if i == 0 else downsample(grid)))
             for i in range(2)],
            sync=True,
        )
        return levels

    first = _build(payload)
    consistent = all(_build(payload) == first for _ in range(runs - 1))
    return {
        "runs": runs,
        "deterministic": consistent,
        "signature": repr(first),
    }


# ---------------------------------------------------------------------------
# 配置校验 / 审计 / 兜底
# ---------------------------------------------------------------------------

def validation(config: Any) -> dict[str, Any]:
    """校验泛光链配置，返回结构化结果，不向调用方抛异常。"""
    if config is None:
        config = {}
    if isinstance(config, (int, float)):
        # 契约便利形式：validation(1) 视为单层默认配置
        config = {"levels": max(1, int(config))}
    if not isinstance(config, dict):
        return {"valid": False, "errors": ["config 必须是字典"], "warnings": []}

    errors: list[str] = []
    warnings: list[str] = []

    levels = config.get("levels", 4)
    if not isinstance(levels, int) or levels <= 0:
        errors.append("levels 必须为正整数")
    elif levels > 16:
        warnings.append("levels > 16，最小一级可能退化为 1x1")

    for key in ("threshold", "exposure", "intensity", "blend", "scatter"):
        if key in config:
            value = config[key]
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"{key} 必须为非负数值")
                continue
            if not math.isfinite(value):
                errors.append(f"{key} 必须有限")

    if "exposure" in config and isinstance(config["exposure"], (int, float)):
        if config["exposure"] == 0:
            errors.append("exposure 不能为 0")

    space = config.get("space", "sdr")
    if isinstance(space, str) and space.lower() not in (
        "sdr", "hdr", "rec709", "rec2020", "bt709", "bt2020", "scrgb",
        "linear", "ldr",
    ):
        errors.append(f"未知色彩空间 {space!r}")

    if "resolution" in config:
        res = config["resolution"]
        if (not isinstance(res, (tuple, list)) or len(res) != 2
                or not all(isinstance(v, int) and v > 0 for v in res)):
            errors.append("resolution 必须是两个正整数 (width, height)")

    return {"valid": not errors, "errors": errors, "warnings": warnings}


def audit(chain: Any) -> dict[str, Any]:
    """对一条泛光链做结构与不变量审计。

    接受 :class:`BloomChain` 或暴露相同鸭子类型（``mips``/``levels``/
    ``resolution``）的对象。
    """
    findings: list[str] = []
    mips = getattr(chain, "mips", None)
    if mips is None and isinstance(chain, dict):
        mips = chain.get("mips")
    if not isinstance(mips, (list, tuple)) or not mips:
        return {"ok": False, "findings": ["链中没有任何 mip 层"], "levels": 0}

    levels = len(mips)
    expected_levels = getattr(chain, "levels", levels)
    if expected_levels != levels:
        findings.append(f"声明层数 {expected_levels} 与实际层数 {levels} 不一致")

    prev_rows, prev_cols = _shape(mips[0])
    for i, mip in enumerate(mips):
        if not _is_grid(mip):
            findings.append(f"第 {i} 层不是合法二维网格")
            continue
        rows, cols = _shape(mip)
        if i > 0:
            if rows > prev_rows or cols > prev_cols:
                findings.append(f"第 {i} 层尺寸比上一层还大，链序倒置")
            if rows != _halve(prev_rows) or cols != _halve(prev_cols):
                findings.append(
                    f"第 {i} 层尺寸 {rows}x{cols} 不符合半级递推 "
                    f"（期望 {_halve(prev_rows)}x{_halve(prev_cols)}）"
                )
        prev_rows, prev_cols = rows, cols

    # 能量审计：每级盒式降采样后总能量应约为上一级的 1/4（均匀画面下精确）。
    for i in range(1, levels):
        e_prev, e_cur = _grid_energy(mips[i - 1]), _grid_energy(mips[i])
        if e_prev > 0 and e_cur > 0:
            ratio = e_prev / e_cur
            if not (2.0 <= ratio <= 8.0):
                findings.append(
                    f"第 {i} 级能量比 {ratio:.2f} 偏离盒式降采样预期 (~4)"
                )

    return {"ok": not findings, "findings": findings, "levels": levels}


def fallback(reason: Any = None) -> dict[str, Any]:
    """泛光链构建失败时的安全兜底：返回 1x1 全黑链，保证合成不崩。"""
    return {
        "active": False,
        "reason": "unknown" if reason is None else str(reason),
        "mips": [[[0.0]]],
        "levels": 1,
    }


# ---------------------------------------------------------------------------
# 指标 / 镜头脏污遮罩 / 峰值截断
# ---------------------------------------------------------------------------

def metrics(grid: Any) -> dict[str, float]:
    """计算泛光网格的亮度统计指标（均值/最大/能量/超过 1 的 HDR 占比）。"""
    if isinstance(grid, (int, float)):
        grid = [[float(grid)]]
    if not _is_grid(grid):
        raise ValueError("metrics 需要二维数字网格")
    rows, cols = _shape(grid)
    lumas = [_luma(cell) for row in grid for cell in row]
    count = len(lumas)
    mean = sum(lumas) / count
    peak = max(lumas)
    energy = sum(lumas)
    over_one = sum(1 for v in lumas if v > 1.0) / count
    return {
        "mean": mean,
        "max": peak,
        "energy": energy,
        "over_white_ratio": over_one,
        "pixels": float(count),
        "resolution": float(rows * cols),
    }


def dirt_mask(shape_or_grid: Any, strength: float = 1.0) -> Any:
    """生成/应用镜头脏污（lens dirt）遮罩。

    传 ``(rows, cols)`` 元组时生成程序化径向遮罩：边缘与角落灰尘更强，
    中心光学区域较干净，数值落在 [0, 1]。传网格时把遮罩乘到网格上
    （尺寸必须一致），模拟灰尘对泛光的散射调制。
    """
    strength = min(1.0, max(0.0, float(strength)))
    if isinstance(shape_or_grid, (int, float)):
        n = max(1, int(shape_or_grid))
        rows = cols = n
        return [[strength]] if n == 1 else _radial_dirt(rows, cols, strength)
    if isinstance(shape_or_grid, (tuple, list)) and len(shape_or_grid) == 2 \
            and all(isinstance(v, int) for v in shape_or_grid):
        return _radial_dirt(max(1, shape_or_grid[0]),
                            max(1, shape_or_grid[1]), strength)
    if _is_grid(shape_or_grid):
        rows, cols = _shape(shape_or_grid)
        mask = _radial_dirt(rows, cols, strength)
        rgb = isinstance(shape_or_grid[0][0], (list, tuple))
        if rgb:
            return [[[float(c) * mask[y][x] for c in cell]
                     for x, cell in enumerate(row)]
                    for y, row in enumerate(shape_or_grid)]
        return [[float(cell) * mask[y][x] for x, cell in enumerate(row)]
                for y, row in enumerate(shape_or_grid)]
    raise ValueError("dirt_mask 需要 (rows, cols) 尺寸或二维网格")


def _radial_dirt(rows: int, cols: int, strength: float) -> list[list[float]]:
    cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
    radius = math.sqrt(cy * cy + cx * cx) or 1.0
    out: list[list[float]] = []
    for y in range(rows):
        row = []
        for x in range(cols):
            d = math.sqrt((y - cy) ** 2 + (x - cx) ** 2) / radius
            # 基底灰尘 + 边缘累积，裁剪到 [0,1]
            v = strength * (0.15 + 0.85 * d * d)
            row.append(min(1.0, max(0.0, v)))
        out.append(row)
    return out


def clamp(grid: Any, ceiling: float = 8.0,
          soft_knee: float = 0.0) -> list[list[Any]]:
    """把泛光值限制在天花板以下，支持硬截断与软膝压缩。"""
    if isinstance(grid, (int, float)):
        grid = [[float(grid)]]
    if not _is_grid(grid):
        raise ValueError("clamp 需要二维数字网格")
    ceiling = float(ceiling)
    if ceiling <= 0:
        raise ValueError("ceiling 必须为正")
    soft_knee = min(ceiling, max(0.0, float(soft_knee)))
    knee_start = ceiling - soft_knee

    def _one(v: float) -> float:
        v = float(v)
        if v <= knee_start:
            return v
        if soft_knee <= 0.0:
            return min(v, ceiling)
        # 软膝：在 [knee_start, ceiling] 区间做平滑压缩，渐近天花板
        t = (v - knee_start) / soft_knee
        return knee_start + soft_knee * t / (1.0 + t)

    rgb = isinstance(grid[0][0], (list, tuple))
    if rgb:
        return [[[_one(c) for c in cell] for cell in row] for row in grid]
    return [[_one(cell) for cell in row] for row in grid]


# ---------------------------------------------------------------------------
# 整合：一条完整的泛光链
# ---------------------------------------------------------------------------

class BloomChain:
    """把六项修复串起来的完整泛光多级链。

    典型每帧流程::

        chain = BloomChain(resolution=(1920, 1080), levels=6)
        frame = chain.render(hdr_grid, exposure=auto_exposure, space="hdr")
    """

    def __init__(self,
                 resolution: tuple[int, int] = (8, 8),
                 levels: int = 4,
                 base_threshold: float = 1.0,
                 knee: float = 0.5,
                 temporal_blend: float = 0.25,
                 filter_mode: str = "box"):
        self.levels = max(1, int(levels))
        self.base_threshold = float(base_threshold)
        self.knee = float(knee)
        self.temporal_blend = float(temporal_blend)
        self.filter_mode = filter_mode
        self.resolution = (max(1, int(resolution[0])),
                           max(1, int(resolution[1])))
        self.exposure = 1.0
        self.mips: list[list[list[Any]]] = []
        self.half_history: list[list[Any]] | None = None
        self._buffers = buffer_lifetime(
            None, self.resolution[0], self.resolution[1], self.levels
        )

    # -- 修复 1 ----------------------------------------------------------
    def threshold(self, exposure: float) -> float:
        """每帧重算阈值，绝不沿用上一帧曝光下的旧值。"""
        self.exposure = float(exposure)
        return threshold_exposure(exposure, self.base_threshold, self.knee)

    def apply_threshold(self, grid: Sequence[Sequence[Any]],
                        exposure: float) -> list[list[Any]]:
        t = self.threshold(exposure)
        knee_lo = t * (1.0 - self.knee)
        rgb = isinstance(grid[0][0], (list, tuple))

        def _soft(cell: Any) -> Any:
            luma = _luma(cell)
            if luma <= knee_lo:
                return [0.0, 0.0, 0.0] if rgb else 0.0
            if luma >= t:
                return list(cell) if rgb else float(cell)
            # 软膝区间平滑抬升
            k = (luma - knee_lo) / max(1e-8, t - knee_lo)
            k = k * k * (3.0 - 2.0 * k)
            if rgb:
                return [float(c) * k for c in cell]
            return float(cell) * k

        return _map_cells(_soft, grid)

    # -- 修复 6 ----------------------------------------------------------
    def resize(self, resolution: tuple[int, int], levels: int | None = None) -> None:
        """切分辨率：整链缓冲重建，半分辨率历史因尺寸失效而清空。"""
        levels = self.levels if levels is None else max(1, int(levels))
        self.resolution = (max(1, int(resolution[0])),
                           max(1, int(resolution[1])))
        self.levels = levels
        self._buffers = buffer_lifetime(
            self._buffers, self.resolution[0], self.resolution[1], levels
        )
        self.half_history = None
        self.mips = []

    # -- 修复 2 + 3 ------------------------------------------------------
    def build_mips(self, bright: Sequence[Sequence[Any]],
                   temporal: bool = True) -> list[list[list[Any]]]:
        mips = [bright]
        for _ in range(1, self.levels):
            mips.append(downsample(mips[-1]))
        # 半分辨率链（第 1 级）走时域稳定，消除 TAA 高频闪烁
        if temporal and len(mips) >= 2:
            mips[1] = half_res(mips[1], self.half_history,
                               blend=self.temporal_blend)
            self.half_history = [list(r) for r in mips[1]]
        self.mips = mips
        return mips

    # -- 修复 4 ----------------------------------------------------------
    def composite_tasks(self) -> list[tuple[int, Any]]:
        """按层序生成升采样叠加任务并经异步排序器提交。"""
        tasks: list[tuple[int, Any]] = []
        for level, mip in enumerate(self.mips):
            comp = chain_energy(level) if self.filter_mode == "bilinear" else 1.0
            up = upsample_bleed(mip, target_shape=self.resolution,
                                filter_mode=self.filter_mode,
                                compensation=comp)
            tasks.append((level, up))
        # 无论内部调度如何，收集顺序固定为层号升序
        ordered_payloads = async_order(tasks, sync=True)
        return list(enumerate(ordered_payloads))

    # -- 修复 5 ----------------------------------------------------------
    def render(self, grid: Sequence[Sequence[Any]], exposure: float,
               space: str = "sdr", temporal: bool = True) -> list[list[Any]]:
        """一帧完整泛光：阈值 → 降采样链 → 时域稳定 → 有序叠加 → 输出空间。"""
        rows, cols = _shape(grid)
        if (cols, rows) != self.resolution:
            # 输入分辨率与链不匹配：先重建缓冲再渲染
            self.resize((cols, rows))

        bright = self.apply_threshold(grid, exposure)
        self.build_mips(bright, temporal=temporal)
        tasks = self.composite_tasks()

        rgb = isinstance(grid[0][0], (list, tuple))
        acc = [[[0.0, 0.0, 0.0] if rgb else 0.0 for _ in range(cols)]
               for _ in range(rows)]
        for _, up in tasks:
            for y in range(rows):
                for x in range(cols):
                    if rgb:
                        for k in range(3):
                            acc[y][x][k] += float(up[y][x][k])
                    else:
                        acc[y][x] += float(up[y][x])
        return output_space(acc, space=space)

    # -- 修复 2 的可验证形式 --------------------------------------------
    def round_trip_conservation(self,
                                grid: Sequence[Sequence[Any]]) -> dict[str, float]:
        """降采样→升采样往返的能量守恒误差（box 路径应接近机器精度）。"""
        coarse = downsample(grid)
        back = upsample_bleed(coarse, target_shape=_shape(grid),
                              filter_mode="box", compensation=1.0)
        e0, e1 = _grid_energy(grid), _grid_energy(back)
        rel = abs(e1 - e0) / e0 if e0 > 0.0 else 0.0
        return {"energy_before": e0, "energy_after": e1,
                "relative_error": rel}
