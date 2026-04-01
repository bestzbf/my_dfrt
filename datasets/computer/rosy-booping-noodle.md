# D4RT 离线预处理模块实现方案

## Context

为 ScanNet、Co3Dv2、BlendedMVS、MVS-Synth 四个静态场景数据集离线推导缺失的
`normals`、`trajs_2d`、`trajs_3d_world`、`valids`、`visibs` 五个模态。

这四个数据集均已有 depth + intrinsics + extrinsics（全为 w2c [T,4,4]），
可通过几何推导精确计算，无需训练任何模型。

代码保存在 `/data2/d4rt/code/datasets/computer/`。

---

## 目录结构

```
/data2/d4rt/code/datasets/computer/
    __init__.py
    depth_to_normals.py   # 核心算法：depth + K → normal map
    depth_to_tracks.py    # 核心算法：depth + K + E → tracks
    run_scannet.py        # ScanNet 预处理脚本
    run_co3dv2.py         # Co3Dv2 预处理脚本
    run_blendedmvs.py     # BlendedMVS 预处理脚本
    run_mvssynth.py       # MVS-Synth 预处理脚本
```

---

## 模块1：`depth_to_normals.py`

### API

```python
def compute_normals(
    depth: np.ndarray,        # [H,W] float32，无效值为 0/nan/inf
    K: np.ndarray,            # [3,3] intrinsics
) -> np.ndarray:              # [H,W,3] float32，单位法向量（相机坐标系），无效处为 [0,0,0]
```

### 算法步骤

1. **有效 mask**：`valid = (depth > 0) & np.isfinite(depth)`
2. **反投影为 3D 点云**（相机坐标系）：
   - 构建像素坐标网格 `(u, v)`，`u` 为列，`v` 为行
   - `X = (u - cx) * depth / fx`
   - `Y = (v - cy) * depth / fy`
   - `Z = depth`
   - 得到 `pts [H,W,3]`，无效处置 0
3. **相邻像素叉积求法线**：
   - `du = pts[v, u+1] - pts[v, u-1]`（水平差分，列方向）
   - `dv = pts[v+1, u] - pts[v-1, u]`（垂直差分，行方向）
   - `normal = cross(du, dv)`，归一化为单位向量
   - 边界处（index 越界）用单侧差分
4. **无效处理**：
   - 任一邻居无效（depth<=0 或 nan）时，该像素 normal = `[0,0,0]`
   - 法线指向相机方向（Z 分量应为负，如果为正则翻转）
5. **返回** `[H,W,3]` float32

### 关键注意事项

- 用 `np.gradient` 或手动差分，不要用 cv2.Sobel（Sobel 不感知 depth 无效值）
- 无效邻居的差分向量应置零，不参与叉积
- 法线在相机坐标系，后续转世界系时 adapter 乘 `R = E[:3,:3].T`（w2c 的转置）

---

## 模块2：`depth_to_tracks.py`

### API

```python
def compute_tracks(
    depths: list[np.ndarray],     # [T][H,W] float32
    intrinsics: np.ndarray,       # [T,3,3]
    extrinsics: np.ndarray,       # [T,4,4] w2c
    num_points: int = 8000,
    boundary_ratio: float = 0.3,
    depth_consistency_thresh: float = 0.05,  # 相对误差阈值（5%）
    rng_seed: int = 42,
) -> dict:
    # 返回：
    # {
    #   "trajs_2d":       [T,N,2]  float32，像素坐标 (x=列, y=行)
    #   "trajs_3d_world": [T,N,3]  float32，世界坐标
    #   "valids":         [T,N]    bool
    #   "visibs":         [T,N]    bool   （静态场景 = valids）
    # }
```

### 辅助函数

```python
def unproject(
    uv: np.ndarray,     # [N,2] (x=col, y=row)
    depth: np.ndarray,  # [N] 对应像素的depth值
    K: np.ndarray,      # [3,3]
) -> np.ndarray:        # [N,3] 相机坐标系

def project(
    pts_cam: np.ndarray,  # [N,3] 相机坐标系
    K: np.ndarray,        # [3,3]
) -> tuple[np.ndarray, np.ndarray]:  # uv [N,2], depth [N]

def depth_boundary_mask(
    depth: np.ndarray,   # [H,W]
    percentile: float = 85.0,
) -> np.ndarray:         # [H,W] bool
```

### 算法步骤（策略A，静态场景）

1. **选参考帧**：第 0 帧（或 depth 有效像素最多的帧）
2. **计算参考帧 depth 边界 mask**（Sobel，同 query_builder 的逻辑）
3. **采 N 个 source 2D 点**：
   - `n_boundary = int(N * boundary_ratio)`，`n_uniform = N - n_boundary`
   - 从边界有效像素中随机采 `n_boundary` 个（不足则全取）
   - 从全部有效像素（depth>0, isfinite）中随机采 `n_uniform` 个
   - 合并，总计 N 个点（不足 N 时用均匀采样补足）
4. **反投影到世界坐标**：
   - 取参考帧 depth 值：`d0[i] = depth_0[v_i, u_i]`
   - 相机坐标：`P_cam0 = unproject(uv0, d0, K[0])`   `[N,3]`
   - 世界坐标：`P_world = inv(E[0]) @ homogeneous(P_cam0)`   `[N,3]`
   - `inv(E[0])` 即 c2w = `E[0]^{-1}` = `[R^T | -R^T t]`
5. **逐帧投影**：对每帧 t：
   - `P_cam_t = E[t] @ homogeneous(P_world)`   `[N,4] → [N,3]`
   - 投影：`uv_t, z_t = project(P_cam_t, K[t])`
   - **in_bounds**：`0 <= u < W and 0 <= v < H`
   - **depth_valid**：`depth_t[v_t, u_t] > 0 and isfinite`（双线性/最近邻采样）
   - **depth_consistent**：`|depth_t[v_t, u_t] - z_t| / z_t < depth_consistency_thresh`
   - `valids[t,i] = in_bounds & depth_valid & depth_consistent`
   - `visibs[t,i] = valids[t,i]`（静态场景假设，无遮挡检测）
6. **trajs_3d_world** 对静态场景每帧相同：广播 `P_world` 为 `[T,N,3]`

### 关键注意事项

- `E` 是 w2c，`inv(E) = E^{-1}`，即 `R^T` 和 `-R^T t`，**不要用 `np.linalg.inv`**（数值不稳定），直接用：
  ```python
  R = E[:3, :3];  t = E[:3, 3]
  R_inv = R.T;    t_inv = -R.T @ t
  ```
- 深度一致性检查中，采样 `depth_t[v_t, u_t]` 用**最近邻**（round），避免 bilinear 引入深度误差
- `z_t = P_cam_t[:,2]`，必须 `> 0`（相机前方）才有效
- depth 单位问题：Co3Dv2 是 scene units，与外参一致，无需转换；其他三个是米，也与外参一致 —— **无需任何单位转换**

---

## 缓存格式

每个 sequence 对应一个 `.npz` 文件：

```
<output_root>/<sequence_name>/precomputed.npz
    normals:         [T, H, W, 3]  float32  （相机坐标系）
    trajs_2d:        [T, N, 2]     float32  （像素坐标）
    trajs_3d_world:  [T, N, 3]     float32  （世界坐标）
    valids:          [T, N]        bool
    visibs:          [T, N]        bool
    ref_frame:       int           （参考帧索引，通常 0）
    num_frames:      int
    num_points:      int           （实际采样点数，可能 < 8000）
```

`T` = 该 sequence 的**全部帧数**（不是 clip_len）。
adapter 在 `load_clip` 时按 `frame_indices` 切片。

---

## run 脚本结构（四个脚本共用同一模式）

```python
# 命令行参数
parser.add_argument('--root', required=True)
parser.add_argument('--output-root', default=None)  # 默认写回 root 内
parser.add_argument('--num-points', type=int, default=8000)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--overwrite', action='store_true')

# 主流程
for seq in tqdm(sequences):
    out_path = output_root / seq / 'precomputed.npz'
    if out_path.exists() and not args.overwrite:
        continue
    try:
        # 1. 用 adapter 加载该 sequence 全部帧的 depths/intrinsics/extrinsics
        # 2. compute_normals 逐帧计算
        # 3. compute_tracks 计算 tracks
        # 4. np.savez_compressed 写出
    except Exception as e:
        print(f"[SKIP] {seq}: {e}")
```

- 用 `multiprocessing.Pool` 或 `concurrent.futures.ProcessPoolExecutor` 并行
- 每个 sequence 独立处理，失败不影响其他
- 用 `np.savez_compressed` 节省磁盘（normals 压缩后约 1/3）

---

## Adapter 集成（4个文件各改一处）

### 修改点（以 ScanNetAdapter 为例）

```python
# __init__ 新增参数
def __init__(self, root, ..., precompute_root=None):
    ...
    self.precompute_root = Path(precompute_root) if precompute_root else None

# load_clip 末尾，UnifiedClip 构建之前插入
normals, trajs_2d, trajs_3d, valids, visibs = None, None, None, None, None
if self.precompute_root is not None:
    cache = self._load_precomputed(sequence_name, frame_indices)
    if cache is not None:
        normals       = [cache['normals'][i] for i in frame_indices]
        trajs_2d      = cache['trajs_2d'][frame_indices]
        trajs_3d      = cache['trajs_3d_world'][frame_indices]
        valids        = cache['valids'][frame_indices]
        visibs        = cache['visibs'][frame_indices]

# metadata 更新
"has_tracks":  trajs_3d is not None,
"has_normals": normals is not None,
```

```python
def _load_precomputed(self, sequence_name, frame_indices):
    path = self.precompute_root / sequence_name / 'precomputed.npz'
    if not path.exists():
        return None
    data = np.load(path)
    # 检查帧索引是否在范围内
    if max(frame_indices) >= data['num_frames']:
        return None
    return data
```

---

## 需要修改的文件

| 文件 | 修改内容 |
|------|---------|
| `computer/__init__.py` | 新建，空文件 |
| `computer/depth_to_normals.py` | 新建，核心算法 |
| `computer/depth_to_tracks.py` | 新建，核心算法 |
| `computer/run_scannet.py` | 新建，预处理脚本 |
| `computer/run_co3dv2.py` | 新建，预处理脚本 |
| `computer/run_blendedmvs.py` | 新建，预处理脚本 |
| `computer/run_mvssynth.py` | 新建，预处理脚本 |
| `adapters/scannet.py` | 加 precompute_root 参数 + _load_precomputed |
| `adapters/co3dv2.py` | 同上 |
| `adapters/blendedmvs.py` | 同上 |
| `adapters/mvssynth.py` | 同上 |

---

## 验证方法

```bash
cd /data2/d4rt/code/datasets/computer

# 对单个 sequence 跑预处理
python run_scannet.py \
    --root /data2/d4rt/datasets/scannet/scannet \
    --output-root /data2/d4rt/datasets/scannet/scannet \
    --num-points 8000 --workers 1

# 验证缓存文件
python -c "
import numpy as np
d = np.load('/data2/d4rt/datasets/scannet/scannet/scene0030_00/precomputed.npz')
print({k: (v.shape, v.dtype) for k, v in d.items()})
"

# 验证 adapter 集成
python -c "
import sys; sys.path.insert(0, '/data2/d4rt/code')
from datasets.registry import create_adapter
a = create_adapter('scannet',
    root='/data2/d4rt/datasets/scannet/scannet',
    precompute_root='/data2/d4rt/datasets/scannet/scannet')
clip = a.load_clip('scene0030_00', list(range(8)))
print('has_tracks:', clip.metadata['has_tracks'])
print('trajs_2d:', clip.trajs_2d.shape)
print('normals:', clip.normals[0].shape)
"
```
