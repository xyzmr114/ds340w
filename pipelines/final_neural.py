"""Final-report neural forecasters for NDTI residue cover: ConvLSTM and the official Earthformer
(CuboidTransformerModel, amazon-science/earth-forecasting-transformer, ext/earthformer @ 7732b03),
trained under the full EXPERIMENT_PROTOCOL.md split on the v2 cubes (local_data/harvest20v2).

Task (identical to ndti_eval_local.py): issue dekads t = 5..10, input window t-5..t (6 dekads),
leads 1, 2, 3. Target = NDTI at t+L on cropland, clear at t+L, NDVI < 0.30 at t+L; a pixel is scored
only if it has >= 1 clear NDTI observation in the input window. The test mask, target and persistence
anchor come from ndti_eval_local.target / .features / .load / .crop_mask (E.CUBES pointed at the
v2 cubes), so every model scored by final_eval.py sees exactly the same pixels.

Splits (no random patch split anywhere):
  train = plan["train"] blocks, seasons 2018-2022
  val   = plan["val"] blocks, season 2023      (early stopping + best-checkpoint selection only)
  test  = plan["test"] blocks, seasons 2024, 2025 (touched once, after training)
Patches: 25 non-overlapping 128 x 128 tiles per 640 x 640 window, issues t = 5..10.
Training/validation samples with zero scored target pixels (all three leads) are dropped.

Inputs per dekad (9 channels): ndti, ndvi, red, nir08, swir16, swir22 (reflectance = DN / 1e4), each
forward-filled within the 6-dekad window only (pixels with no clear value yet = training mean, i.e. 0
after standardisation), clear flag, crop mask, dekad index / 13. Every channel is standardised with
mean / std from training samples only (continuous channels over observed values).

Storage: each cube is re-tiled once into a float16 memory-mapped cache
local_data/neural_cache_v2/blk_<bbb>_<Y>.npy, shape (25 tiles, 14 dekads, 6 vars, 128, 128), NaN =
not clear (the clear rule of ndti_eval_local.load; bands masked with it), plus crop_<bbb>.npy.
68.8 MB per cube; full run = 160 train + 10 val + 64 test cubes = 16.1 GB on disk, read lazily
(one sample = one contiguous 1.8 MB slice). Forward fill, masks and normalisation run on the GPU.

Models (P50 = last forward-filled input NDTI + residual; P10 / P90 = P50 -/+ softplus: never crossing)
  earthformer  SEVIR configuration (cfg_sevir.yaml: base_units 128, enc/dec depth [1,1], 4 heads,
               8 global vectors, axial self-attention, cross_1x1, 3-layer stack-conv stem [16,64,128])
               with stem downscale [1,2,2] instead of [3,2,2]: SEVIR's 384 px frame -> 32 x 32 latent;
               our 128 px tile -> the same 32 x 32 latent (each token = 80 m). bf16 autocast,
               gradient checkpointing (checkpoint_level, default 2).
  convlstm     3 (or --convlstm-layers 2) stacked ConvLSTM cells, hidden 64, 3x3 kernels, full 128 x 128 resolution, 1x1 head.
Training: masked pinball loss (mean over scored pixels and P10/P50/P90), AdamW, linear warm-up then
cosine decay to 1e-3 x lr over an explicit --epochs budget (the schedule always completes when the
budget is reached: fixes the Week 9 OneCycle sized for 40 epochs but cut at ~14), random rot90/flip,
grad clip 1, early stopping on validation pinball (patience), best checkpoint kept and used for test.
Per-epoch resumable checkpoints (--resume).

Outputs (results_v2/final/, or results_v2/final/<--tag>/):
  neural_rows.csv    final_eval.py rows.csv schema (FIELDS; split = "test"), model = "<name>_s<seed>";
                     scored rows asserted identical to final_eval.extract's for every unit, so
                     pd.concat([rows.csv, neural_rows.csv]) passes final_analyze's same-n check
  neural_runs.json   per run: config, hyper-parameters, epochs, minutes, params, peak GPU memory,
                     history, normalisation, Earthformer commit, test summary vs persistence
  neural_ckpt/<run>_best.pt, <run>_last.pt
  neural_preds/<run>.npz   key "<block>_<season>_<issue>_<lead>" -> (n, 3) P10/P50/P90 on the scored
                     pixels of that (block, season, issue, lead), in np.ravel order of the 640 x 640
                     mask (has & isfinite(target)) -- rescoring needs only the cubes

Usage:
  python final_neural.py --models convlstm earthformer --seeds 340 341 342 --epochs 12 --resume --skip-done
  python final_neural.py --smoke          # 1 train / 1 val / 2 test cubes, 1 epoch, seed 340
  python final_neural.py --estimate       # data sizes only
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def replace_retry(src, dst, tries=60, wait=2.0):
    """os.replace that waits out short Windows file locks (antivirus or indexer scanning a new file, a reader
    holding the target open). Earthformer s341 lost its test outputs to WinError 32 on 2026-11-06."""
    for i in range(tries):
        try:
            return os.replace(src, dst)
        except PermissionError as e:
            if i == tries - 1:
                raise
            print(f"  replace {os.path.basename(dst)} blocked ({e.winerror if hasattr(e, 'winerror') else e}); retry {i + 1}", flush=True)
            time.sleep(wait)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
EF_ROOT = os.path.join(ROOT, "ext", "earthformer")
sys.path.insert(0, os.path.join(EF_ROOT, "src"))
import ndti_eval_local as E  # noqa: E402

E.CUBES = os.path.join(ROOT, "local_data", "harvest20v2")      # v2 cubes; E.load / E.crop_mask read here
import final_eval as FE  # noqa: E402  (row schema + scored-row reference; reads harvest20v2 itself)
from mh_grid import superblock  # noqa: E402

assert os.path.normcase(os.path.normpath(FE.CUBES)) == os.path.normcase(os.path.normpath(E.CUBES))
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel  # noqa: E402

warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad.*")

CUBES = E.CUBES
CACHE = os.path.join(ROOT, "local_data", "neural_cache_v2")
PLAN = os.path.join(ROOT, "results_v2", "ndti_local", "plan.json")
OUT_BASE = os.path.join(ROOT, "results_v2", "final")
P, NT, W = 128, 25, 640                       # tile size, tiles per window, window size
T_IN, LEADS, QS = E.T_IN, tuple(E.LEADS), tuple(E.QS)
N_DEK = 14
ISSUES = list(range(T_IN - 1, N_DEK - max(LEADS)))          # 5..10
SPAN = T_IN + max(LEADS)                                     # dekads read per sample (window + targets)
TRAIN_Y, VAL_Y, TEST_Y = list(E.TRAIN), list(E.VAL), list(E.TEST)
BANDS = ["red", "nir08", "swir16", "swir22"]
VARS = ["ndti", "ndvi"] + BANDS
INPUTS = [v + "_ffill" for v in VARS] + ["clear", "crop", "dekad"]
C_IN, N_CONT = len(INPUTS), len(VARS)
# final_eval.py rows.csv schema (score_bs keys, with "split" inserted at position 1 as final_eval.main does)
FIELDS = ["model", "split", "block", "season", "issue", "lead", "n", "sae", "sse", "se", "spin", "cover", "below",
          "nfb", "lastgreen", "clear_rate", "cropfrac", "wc_cropfrac", "region", "superblock", "ndvi_peak",
          "season_clear", "clim_years", "seas_prev"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================ cache / data
def cube_src(b, y):
    return os.path.join(CUBES, f"blk_{b:03d}_{y}.npz")


def cache_path(b, y):
    return os.path.join(CACHE, f"blk_{b:03d}_{y}.npy")


def tiles(a):
    """(..., 640, 640) -> (25, ..., 128, 128), tile k = (row k // 5, col k % 5)."""
    lead = a.shape[:-2]
    n = W // P
    a = a.reshape(lead + (n, P, n, P))
    nd = len(lead)
    a = a.transpose((nd, nd + 2) + tuple(range(nd)) + (nd + 1, nd + 3))
    return a.reshape((NT,) + lead + (P, P))


def ensure_cache(b, y):
    """Build (once) the float16 tiled cache of cube (b, y). Returns meta dict or None if no cube."""
    src = cube_src(b, y)
    if not os.path.exists(src):
        return None
    out, mpath = cache_path(b, y), cache_path(b, y)[:-4] + ".json"
    st = os.stat(src)
    key = [int(st.st_size), int(st.st_mtime)]
    if os.path.exists(out) and os.path.exists(mpath):
        meta = json.load(open(mpath))
        if meta.get("src") == key:
            return meta
    d = E.load(b, y)                           # NaN where not clear (identical rule to final_eval)
    if d is None:
        return None
    ndti, ndvi = d
    if ndti.shape != (N_DEK, W, W):
        print(f"  skip {b} {y}: shape {ndti.shape}", flush=True)
        return None
    z = np.load(src)
    clear = np.isfinite(ndti)
    arr = np.empty((N_DEK, len(VARS), W, W), np.float16)
    arr[:, 0], arr[:, 1] = ndti, ndvi
    for i, k in enumerate(BANDS):
        dn = z[k]
        arr[:, 2 + i] = np.where(clear & (dn > 0), dn.astype(np.float32) / 1e4, np.nan)
    crop = E.crop_mask(b)
    cpath = os.path.join(CACHE, f"crop_{b:03d}.npy")
    if not os.path.exists(cpath):
        tmp = cpath[:-4] + ".tmp.npy"
        np.save(tmp, tiles(crop))
        replace_retry(tmp, cpath)
    # scored target pixels per (tile, issue), summed over leads (used to drop empty train samples)
    nvalid = np.zeros((NT, len(ISSUES)), np.int64)
    for j, t in enumerate(ISSUES):
        has = np.isfinite(ndti[t - T_IN + 1:t + 1]).any(0)
        for L in LEADS:
            ok = has & np.isfinite(E.target(ndti, ndvi, crop, t + L))
            nvalid[:, j] += tiles(ok).reshape(NT, -1).sum(1)
    tmp = out[:-4] + ".tmp.npy"
    np.save(tmp, np.ascontiguousarray(tiles(arr)))       # (25, 14, 6, P, P)
    replace_retry(tmp, out)
    meta = {"src": key, "nvalid": nvalid.tolist(), "block": b, "season": y}
    json.dump(meta, open(mpath, "w"))
    return meta


class Store:
    """Lazy, memory-mapped access to cached cubes. index rows = (cube id, tile, issue t)."""

    def __init__(self):
        self.cubes, self.mm, self.crop, self.meta = [], [], {}, []

    def add(self, b, y):
        meta = ensure_cache(b, y)
        if meta is None:
            return None
        cid = len(self.cubes)
        self.cubes.append((b, y))
        self.mm.append(np.load(cache_path(b, y), mmap_mode="r"))
        if b not in self.crop:
            self.crop[b] = np.load(os.path.join(CACHE, f"crop_{b:03d}.npy"))
        self.meta.append(meta)
        return cid

    def index(self, cids, drop_empty):
        rows = []
        for c in cids:
            nv = np.array(self.meta[c]["nvalid"])
            for p in range(NT):
                for j, t in enumerate(ISSUES):
                    if drop_empty and nv[p, j] == 0:
                        continue
                    rows.append((c, p, t))
        return np.array(rows, np.int64).reshape(-1, 3)

    def fetch(self, rows):
        raw = np.stack([self.mm[c][p, t - T_IN + 1:t - T_IN + 1 + SPAN] for c, p, t in rows])
        crop = np.stack([self.crop[self.cubes[c][0]][p] for c, p, _ in rows])
        return torch.from_numpy(raw), torch.from_numpy(crop), torch.from_numpy(rows[:, 2].copy())


def batches(store, idx, bs, shuffle, rng, ahead=4, workers=3):
    """Yields CPU tensor batches; reads run in a thread pool a few batches ahead."""
    order = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
    chunks = [idx[np.sort(order[i:i + bs])] for i in range(0, len(order), bs)]
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(store.fetch, ch) for ch in chunks[:ahead]]
        for k in range(len(chunks)):
            if k + ahead < len(chunks):
                futs.append(ex.submit(store.fetch, chunks[k + ahead]))
            yield futs[k].result()
            futs[k] = None


def assemble(raw, crop, t):
    """GPU. raw (B, SPAN, 6, P, P) f16 dekads t-5..t+3; crop (B, P, P) bool; t (B,) issue.
    Returns x (B, T_IN, C_IN, P, P) unnormalised with NaN in continuous channels where no clear value
    yet in the window, y (B, 3, P, P), m (B, 3, P, P) scored mask, last (B, P, P) ffilled NDTI at t."""
    raw = raw.to(DEV, non_blocking=True).float()
    crop = crop.to(DEV, non_blocking=True)
    t = t.to(DEV)
    win = raw[:, :T_IN]
    clear = torch.isfinite(win[:, :, 0])                    # (B, T, P, P)
    last = torch.full_like(win[:, 0], float("nan"))
    ff = []
    for i in range(T_IN):
        last = torch.where(torch.isfinite(win[:, i]), win[:, i], last)
        ff.append(last)
    ff = torch.stack(ff, 1)                                 # (B, T, 6, P, P)
    has = clear.any(1)
    B = raw.shape[0]
    dek = ((t[:, None] - T_IN + 1 + torch.arange(T_IN, device=DEV)[None]).float() / 13.0)
    x = torch.cat([ff, clear[:, :, None].float(),
                   crop[:, None, None].float().expand(B, T_IN, 1, P, P),
                   dek[:, :, None, None, None].expand(B, T_IN, 1, P, P)], 2)
    tg = raw[:, T_IN:]
    tn, tv = tg[:, :, 0], tg[:, :, 1]
    m = crop[:, None] & torch.isfinite(tn) & (tv < E.NDVI_MAX) & has[:, None]
    return x, torch.nan_to_num(tn, nan=0.0), m, torch.nan_to_num(ff[:, -1, 0], nan=0.0)


def normalise(x, mu, sd):
    return torch.nan_to_num((x - mu) / sd, nan=0.0)


def norm_stats(store, idx, rng, n=384):
    """Per-channel mean/std from training samples only (continuous channels: observed values)."""
    sel = idx[np.sort(rng.choice(len(idx), min(n, len(idx)), replace=False))]
    s = torch.zeros(C_IN, dtype=torch.float64, device=DEV)
    s2, cnt = torch.zeros_like(s), torch.zeros_like(s)
    for i in range(0, len(sel), 32):
        x, _, _, _ = assemble(*store.fetch(sel[i:i + 32]))
        xc = x.transpose(0, 2).reshape(C_IN, -1).double()  # (C, ...)
        ok = torch.isfinite(xc)
        xz = torch.where(ok, xc, torch.zeros_like(xc))
        s += xz.sum(1); s2 += (xz ** 2).sum(1); cnt += ok.sum(1)
    mu = s / cnt.clamp(min=1)
    sd = torch.sqrt((s2 / cnt.clamp(min=1) - mu ** 2).clamp(min=0)) + 1e-6
    sd = torch.where(sd < 1e-4, torch.ones_like(sd), sd)   # constant channel (e.g. one-cube smoke)
    return mu.float().cpu().numpy(), sd.float().cpu().numpy()


# ============================================================================ models
def quantile_head(raw, last):
    """raw (B, 3 leads, H, W, 3) -> P10, P50, P90 (non-crossing), P50 = last ffilled NDTI + residual."""
    med = last[:, None] + raw[..., 0]
    return torch.stack([med - F.softplus(raw[..., 1]), med, med + F.softplus(raw[..., 2])], -1)


def earthformer_cfg(ckpt_level=2):
    """cfg_sevir.yaml 'model' block (ext/earthformer/scripts/cuboid_transformer/sevir), patterns
    expanded as train_cuboid_sevir.py does; shapes and stem downscale adapted to 128 px tiles."""
    return dict(
        input_shape=[T_IN, P, P, C_IN], target_shape=[len(LEADS), P, P, 3],
        base_units=128, block_units=None, scale_alpha=1.0,
        enc_depth=[1, 1], dec_depth=[1, 1], enc_use_inter_ffn=True, dec_use_inter_ffn=True,
        dec_hierarchical_pos_embed=False, downsample=2, downsample_type="patch_merge",
        upsample_type="upsample", num_global_vectors=8, use_dec_self_global=False,
        dec_self_update_global=True, use_dec_cross_global=False, use_global_vector_ffn=False,
        use_global_self_attn=True, separate_global_qkv=True, global_dim_ratio=1,
        enc_attn_patterns=["axial"] * 2, dec_self_attn_patterns=["axial"] * 2,
        dec_cross_attn_patterns=["cross_1x1"] * 2, dec_cross_last_n_frames=None,
        attn_drop=0.1, proj_drop=0.1, ffn_drop=0.1, num_heads=4, ffn_activation="gelu",
        gated_ffn=False, norm_layer="layer_norm", padding_type="zeros", pos_embed_type="t+h+w",
        use_relative_pos=True, self_attn_use_final_proj=True, dec_use_first_self_attn=False,
        z_init_method="zeros", checkpoint_level=ckpt_level, initial_downsample_type="stack_conv",
        initial_downsample_activation="leaky", initial_downsample_stack_conv_num_layers=3,
        initial_downsample_stack_conv_dim_list=[16, 64, 128],
        initial_downsample_stack_conv_downscale_list=[1, 2, 2],
        initial_downsample_stack_conv_num_conv_list=[2, 2, 2],
        attn_linear_init_mode="0", ffn_linear_init_mode="0", conv_init_mode="0",
        down_up_linear_init_mode="0", norm_init_mode="0")


class EarthformerQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = CuboidTransformerModel(**cfg)

    def forward(self, x, last):                 # x (B, T, C, H, W)
        return quantile_head(self.net(x.permute(0, 1, 3, 4, 2).contiguous()), last)


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hid, k):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(cin + hid, 4 * hid, k, padding=k // 2)

    def forward(self, x, h, c):
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class ConvLSTMQ(nn.Module):
    def __init__(self, layers=3, hidden=64, kernel=3):
        super().__init__()
        self.cells = nn.ModuleList([ConvLSTMCell(C_IN if i == 0 else hidden, hidden, kernel)
                                    for i in range(layers)])
        self.head = nn.Conv2d(hidden, len(LEADS) * 3, 1)

    def forward(self, x, last):
        B, T, C, H, W_ = x.shape
        hs = [x.new_zeros(B, c.hid, H, W_) for c in self.cells]
        cs = [x.new_zeros(B, c.hid, H, W_) for c in self.cells]
        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.cells):
                hs[i], cs[i] = cell(inp, hs[i], cs[i])
                inp = hs[i]
        y = self.head(hs[-1]).view(B, len(LEADS), 3, H, W_).permute(0, 1, 3, 4, 2)
        return quantile_head(y, last)


# Fixed hyper-parameters (no search on either family; equal budget = zero). Batch 8 for both keeps the
# peak under ~5 GB: on Windows/WDDM, exceeding free VRAM spills to shared memory (~10x slower), not OOM.
# Earthformer: cfg_sevir.yaml trains with lr 1e-3 at total batch 32 and 20% warm-up. We use batch 8, so the
# lr is scaled linearly to 2.5e-4 and warm-up set to 20%. The first final run (seed 340) used lr 1e-3 with
# 10% warm-up and diverged after warm-up (train pinball 0.0088 -> 0.0117 at epoch 3, val 0.0070 -> 0.0103);
# that run is archived in results_v2/final/archive_earthformer_lr1e-3/.
HP = {"earthformer": dict(bs=8, lr=2.5e-4, wd=1e-5, warmup=0.2, patience=6),
      "convlstm": dict(bs=8, lr=1e-3, wd=1e-4, warmup=0.05, patience=6)}
CONVLSTM_CFG = dict(layers=3, hidden=64, kernel=3)


def make_model(name, args):
    if name == "earthformer":
        cfg = earthformer_cfg(args.ckpt_level)
        return EarthformerQ(cfg), cfg
    cfg = dict(CONVLSTM_CFG, layers=args.convlstm_layers)
    return ConvLSTMQ(**cfg), cfg


# ============================================================================ training
def pinball_loss(q, y, m):
    tau = torch.tensor(QS, device=q.device)
    e = y[..., None] - q
    pl = torch.maximum(tau * e, (tau - 1) * e).mean(-1)
    mf = m.float()
    return (pl * mf).sum() / mf.sum().clamp(min=1)


def augment(k, flip, *ts):
    out = []
    for a in ts:
        a = torch.rot90(a, k, (-2, -1))
        out.append(torch.flip(a, (-1,)) if flip else a)
    return out


@torch.no_grad()
def val_pinball(model, store, idx, mu, sd, bs):
    model.eval()
    num = den = 0.0
    for raw, crop, t in batches(store, idx, bs, False, None):
        x, y, m, last = assemble(raw, crop, t)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            q = model(normalise(x, mu, sd), last)
        k = float(m.sum())
        num += float(pinball_loss(q.float(), y, m)) * k
        den += k
    return num / max(den, 1.0)


def lr_lambda(total, warm, floor=1e-3):
    def f(step):
        if step < warm:
            return (step + 1) / warm
        p = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return f


def save_atomic(obj, path):
    """A crash mid-write must not corrupt the checkpoint --resume depends on."""
    torch.save(obj, path + ".tmp")
    replace_retry(path + ".tmp", path)


def train_run(name, seed, args, store, tr_idx, va_idx, ckdir):
    hp = dict(HP[name])
    for k in ("bs", "lr", "patience"):
        if getattr(args, k) is not None:
            hp[k] = getattr(args, k)
    run = f"{name}_s{seed}"
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.backends.cudnn.benchmark = True
    mu, sd = norm_stats(store, tr_idx, np.random.default_rng(340))     # same stats for all seeds
    mu_t = torch.tensor(mu, device=DEV).view(1, 1, C_IN, 1, 1)
    sd_t = torch.tensor(sd, device=DEV).view(1, 1, C_IN, 1, 1)
    model, cfg = make_model(name, args)
    model = model.to(DEV)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    steps_ep = math.ceil(len(tr_idx) / hp["bs"])
    total = steps_ep * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, max(1, int(hp["warmup"] * total))))
    best, bad, hist, start, stopped, best_ep = 1e9, 0, [], 0, "epoch_budget", -1
    last_p, best_p = os.path.join(ckdir, f"{run}_last.pt"), os.path.join(ckdir, f"{run}_best.pt")
    t_spent = 0.0
    if args.resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=DEV, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, bad, hist, start, best_ep, t_spent = ck["best"], ck["bad"], ck["hist"], ck["epoch"] + 1, ck["best_ep"], ck["t_spent"]
        rng = np.random.default_rng(seed + 1000 * start)
        print(f"  [{run}] resumed after epoch {start - 1}", flush=True)
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time() - t_spent
    step_times = []
    for ep in range(start, args.epochs):
        if bad >= hp["patience"]:
            stopped = "early_stopping"; break
        model.train()
        te = time.time()
        tl = []
        for si, (raw, crop, t) in enumerate(batches(store, tr_idx, hp["bs"], True, rng)):
            ts = time.time()
            x, y, m, last = assemble(raw, crop, t)
            if not bool(m.any()):
                sched.step(); continue
            x = normalise(x, mu_t, sd_t)
            x, y, m, last = augment(int(rng.integers(4)), bool(rng.random() < 0.5), x, y, m, last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                q = model(x, last)
            loss = pinball_loss(q.float(), y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl.append(loss.item())
            if si >= 3:
                step_times.append(time.time() - ts)
            if args.max_steps and si + 1 >= args.max_steps:
                break
        tr_sec = time.time() - te
        tv = time.time()
        vl = val_pinball(model, store, va_idx, mu_t, sd_t, hp["bs"] * 2)
        va_sec = time.time() - tv
        hist.append(dict(epoch=ep, train=float(np.mean(tl)) if tl else None, val=vl,
                         lr=float(sched.get_last_lr()[0]), train_sec=tr_sec, val_sec=va_sec))
        el = (time.time() - t0) / 60
        print(f"  [{run}] ep {ep + 1}/{args.epochs} train {hist[-1]['train']:.5f} val {vl:.5f} "
              f"lr {hist[-1]['lr']:.2e} ({tr_sec:.0f}+{va_sec:.0f} s, {el:.1f} min)", flush=True)
        if math.isfinite(vl) and vl < best - 1e-6:
            best, bad, best_ep = vl, 0, ep
            save_atomic({"model": model.state_dict(), "cfg": cfg, "name": name, "seed": seed, "epoch": ep,
                         "val_pinball": vl, "norm_mu": mu.tolist(), "norm_sd": sd.tolist(),
                         "inputs": INPUTS}, best_p)
        else:
            bad += 1
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "best": best, "bad": bad, "hist": hist, "epoch": ep, "best_ep": best_ep,
                     "t_spent": time.time() - t0}, last_p)
        if bad >= hp["patience"]:
            stopped = "early_stopping"; break
        if args.max_hours and el / 60 * (ep + 2) / (ep + 1) > args.max_hours:
            stopped = "time_cap"; break
    if best_ep >= 0 and os.path.exists(best_p):
        ck = torch.load(best_p, map_location=DEV, weights_only=False)
        model.load_state_dict(ck["model"])
    else:
        print(f"  [{run}] WARNING: no finite validation improvement; testing the last weights", flush=True)
        stopped += "_no_best"
    info = dict(run=run, model=name, seed=seed, params=int(nparams), config=cfg, hyper=hp,
                epoch_budget=args.epochs, epochs_run=len(hist), best_epoch=best_ep + 1,
                best_val_pinball=best, stopped=stopped, train_minutes=(time.time() - t0) / 60,
                steps_per_epoch=steps_ep, sec_per_step=float(np.median(step_times)) if step_times else None,
                peak_gpu_gb=(torch.cuda.max_memory_allocated() / 2 ** 30) if DEV == "cuda" else None,
                n_train_samples=int(len(tr_idx)), n_val_samples=int(len(va_idx)),
                norm_mu=mu.tolist(), norm_sd=sd.tolist(), history=hist,
                resumed_after_epoch=(start - 1 if start else None),
                timing_note=("resumed run: train_minutes includes the saved t_spent; sec_per_step and "
                             "peak_gpu_gb cover only the epochs run after resuming") if start else None)
    return model, (mu_t, sd_t), info


# ============================================================================ test scoring
@torch.no_grad()
def predict_window(model, store, cid, t, mu, sd, bs):
    """(3 leads, 640, 640, 3) float32 for issue t of cube cid."""
    model.eval()
    rows = np.array([(cid, p, t) for p in range(NT)], np.int64)
    out = np.empty((NT, len(LEADS), P, P, 3), np.float32)
    for i in range(0, NT, bs):
        x, _, _, last = assemble(*store.fetch(rows[i:i + bs]))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            q = model(normalise(x, mu, sd), last)
        out[i:i + bs] = q.float().cpu().numpy()
    n = W // P
    return out.reshape(n, n, len(LEADS), P, P, 3).transpose(2, 0, 3, 1, 4, 5).reshape(len(LEADS), W, W, 3)


def FE_ndvi_peak(ndvi, crop):
    """final_eval.extract's ndvi_peak (dekad of the peak cropland-median NDVI); copied because
    final_eval computes it only in its 'full' mode, which also builds climatology from 5 cubes."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(crop[None], ndvi, np.nan).reshape(ndvi.shape[0], -1), 1)
    return int(np.nanargmax(med)) if np.isfinite(med).any() else -1


def pinball_np(y, q, tau):
    e = y - q
    return np.maximum(tau * e, (tau - 1) * e)


def score_test(model, norm, store, test_cids, run, plan, bs, pred_dtype):
    """Rows in final_eval.score_bs's schema (FIELDS), on final_eval.extract's exact rows: the scored
    pixel set, its order and the targets are asserted equal to final_eval's for every unit, and the
    per-unit meta columns (clear_rate, lastgreen, wc_cropfrac, ndvi_peak, season_clear, clim_years,
    seas_prev, region, superblock) are computed by final_eval's own code."""
    mu, sd = norm
    rows, preds = [], {}
    agg = {"n": 0, "ae": 0.0, "pin": 0.0, "cover": 0, "below": 0, "ae_pers": 0.0, "pin_pers": 0.0}
    gi = FE.FEATS.index("ndvi_last")
    for cid in test_cids:
        b, y = store.cubes[cid]
        ndti, ndvi = E.load(b, y)
        crop = E.crop_mask(b)
        ref = FE.extract(b, y, "sample", 10 ** 12, 2)          # per huge = every scored pixel, no cache write
        meta, deps = json.loads(str(ref["meta"])), FE.deps_of(b, y, "full")
        peak = FE_ndvi_peak(ndvi, crop)
        bounds = np.concatenate([[0], np.cumsum(ref["units"][:, 2])])
        uj = 0
        for t in ISSUES:
            f, has = E.features(ndti, ndvi, t)
            Q = predict_window(model, store, cid, t, mu, sd, bs)
            for li, L in enumerate(LEADS):
                tg = E.target(ndti, ndvi, crop, t + L)
                ok = (has & np.isfinite(tg)).ravel()
                n = int(ok.sum())
                ut, uL, un = (int(v) for v in ref["units"][uj])
                sl = slice(int(bounds[uj]), int(bounds[uj + 1]))
                uj += 1
                assert (ut, uL, un) == (t, L, n), f"unit mismatch vs final_eval: {b} {y} {(ut, uL, un)} != {(t, L, n)}"
                if n == 0:
                    continue
                yv = tg.ravel()[ok]
                assert np.array_equal(yv, ref["y"][sl]), f"target rows differ from final_eval: {b} {y} {t} {L}"
                p = np.sort(Q[li].reshape(-1, 3)[ok], 1)
                bad = ~np.isfinite(p).all(1)
                if bad.any():                                      # documented fallback, counted in nfb
                    p = np.where(np.isfinite(p), p, np.nanmedian(yv))
                preds[f"{b}_{y}_{t}_{L}"] = p.astype(pred_dtype)
                e = p[:, 1] - yv
                pin = (pinball_np(yv, p[:, 0], .1) + pinball_np(yv, p[:, 1], .5) + pinball_np(yv, p[:, 2], .9)) / 3
                rows.append({"model": run, "split": "test", "block": b, "season": y, "issue": t, "lead": L, "n": n,
                             "sae": float(np.abs(e).sum()), "sse": float((e ** 2).sum()), "se": float(e.sum()),
                             "spin": float(pin.sum()), "cover": int(((yv >= p[:, 0]) & (yv <= p[:, 2])).sum()),
                             "below": int((yv < p[:, 0]).sum()), "nfb": int(bad.sum()),
                             "lastgreen": int((ref["X"][sl, gi] >= E.NDVI_MAX).sum()),
                             "clear_rate": round(meta["clear_rate"][ISSUES.index(t)], 4),
                             "cropfrac": plan["cropfrac"].get(str(b), np.nan),
                             "wc_cropfrac": round(meta["wc_cropfrac"], 4),
                             "region": plan["division"].get(str(b), ""), "superblock": superblock(b),
                             "ndvi_peak": peak, "season_clear": round(meta["season_clear"], 4),
                             "clim_years": len(deps["clim_years"]), "seas_prev": int(deps["prev"])})
                pers = f["ndti_last"].ravel()[ok]
                agg["n"] += n; agg["ae"] += float(np.abs(e).sum()); agg["pin"] += float(pin.sum())
                agg["cover"] += rows[-1]["cover"]; agg["below"] += rows[-1]["below"]
                agg["ae_pers"] += float(np.abs(pers - yv).sum())
                agg["pin_pers"] += float(pinball_np(yv, pers, 0.5).sum())   # mean of 3 equal quantiles
    nn_ = max(agg["n"], 1)
    summ = dict(pixels=agg["n"], mae=agg["ae"] / nn_, pinball=agg["pin"] / nn_, cov80=agg["cover"] / nn_,
                below_p10=agg["below"] / nn_, persistence_mae=agg["ae_pers"] / nn_,
                persistence_pinball=agg["pin_pers"] / nn_)
    return rows, preds, summ


# ============================================================================ main
def parse_cubes(s):
    out = []
    for tok in s.split(","):
        b, y = tok.split(":")
        out.append((int(b), int(y)))
    return out


def ef_commit():
    try:
        return subprocess.check_output(["git", "-C", EF_ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def write_outputs(out, run, rows, info):
    import pandas as pd
    os.makedirs(out, exist_ok=True)
    fields = FIELDS
    ref = os.path.join(OUT_BASE, "rows.csv")
    if os.path.exists(ref):
        have = list(pd.read_csv(ref, nrows=0).columns)
        if have != FIELDS:
            print(f"  WARNING: final_eval rows.csv columns {have} differ from FIELDS {FIELDS}", flush=True)
    assert not rows or list(rows[0]) == FIELDS, (list(rows[0]), FIELDS)
    new = pd.DataFrame([{k: r.get(k, "") for k in fields} for r in rows], columns=fields)
    path = os.path.join(out, "neural_rows.csv")
    if os.path.exists(path):
        old = pd.read_csv(path, keep_default_na=False)
        if list(old.columns) != fields:
            sys.exit(f"{path} has an older row schema {list(old.columns)}; move it aside and rerun")
        old = old[old["model"] != run]
        new = pd.concat([old, new], ignore_index=True)
    tmp = path + ".tmp"
    new.to_csv(tmp, index=False); replace_retry(tmp, path)
    jp = os.path.join(out, "neural_runs.json")
    runs = json.load(open(jp)) if os.path.exists(jp) else {}
    runs[run] = info
    json.dump(runs, open(jp + ".tmp", "w"), indent=1, default=float); replace_retry(jp + ".tmp", jp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["convlstm", "earthformer"], choices=["convlstm", "earthformer"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[340, 341, 342])
    ap.add_argument("--epochs", type=int, default=12, help="epoch budget: the LR schedule spans exactly this")
    ap.add_argument("--convlstm-layers", type=int, default=3, choices=[2, 3])
    ap.add_argument("--allow-missing", action="store_true", help="full run: proceed even if cubes are missing")
    ap.add_argument("--bs", type=int, default=None); ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--ckpt-level", type=int, default=2, help="Earthformer gradient checkpointing (0/1/2)")
    ap.add_argument("--max-hours", type=float, default=0, help="safety cap per run (0 = none)")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: cap steps per epoch")
    ap.add_argument("--resume", action="store_true", help="continue from <run>_last.pt")
    ap.add_argument("--skip-done", action="store_true", help="skip runs already in neural_runs.json")
    ap.add_argument("--pred-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--tag", default="", help="write to results_v2/final/<tag>/ instead")
    ap.add_argument("--train-cubes", default=None, help="override, e.g. 142:2024,176:2024")
    ap.add_argument("--val-cubes", default=None); ap.add_argument("--test-cubes", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run on cubes already present")
    ap.add_argument("--estimate", action="store_true", help="print data sizes and exit")
    args = ap.parse_args()
    plan = json.load(open(PLAN))
    if args.smoke:
        have = sorted(f for f in os.listdir(CUBES) if f.startswith("blk_") and f.endswith(".npz"))
        have = [(int(f[4:7]), int(f[8:12])) for f in have]
        if len(have) < 3:
            sys.exit("smoke test needs >= 3 cubes")
        args.train_cubes = args.train_cubes or f"{have[0][0]}:{have[0][1]}"
        args.val_cubes = args.val_cubes or f"{have[1][0]}:{have[1][1]}"
        args.test_cubes = args.test_cubes or ",".join(f"{b}:{y}" for b, y in have[2:4])
        args.epochs, args.tag = 1 if args.epochs == 12 else args.epochs, args.tag or "smoke"
        args.seeds = args.seeds if args.seeds != [340, 341, 342] else [340]
    tr_c = parse_cubes(args.train_cubes) if args.train_cubes else [(b, y) for b in plan["train"] for y in TRAIN_Y]
    va_c = parse_cubes(args.val_cubes) if args.val_cubes else [(b, y) for b in plan["val"] for y in VAL_Y]
    te_c = parse_cubes(args.test_cubes) if args.test_cubes else [(b, y) for b in plan["test"] for y in TEST_Y]
    # leakage guard: the three block sets are disjoint, and outside --smoke every cube must sit in its
    # protocol split (train blocks x 2018-2022, val blocks x 2023, test blocks x 2024-2025)
    sets = {k: set(plan[k]) for k in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    if not args.smoke:
        for nm, cl, yrs in (("train", tr_c, TRAIN_Y), ("val", va_c, VAL_Y), ("test", te_c, TEST_Y)):
            bad = [(b, y) for b, y in cl if b not in sets[nm] or y not in yrs]
            if bad:
                sys.exit(f"{nm} cubes outside the protocol split: {bad[:5]} (only --smoke may do this)")
    per_cube = NT * N_DEK * len(VARS) * P * P * 2
    print(f"cubes planned: train {len(tr_c)}, val {len(va_c)}, test {len(te_c)}; cache {per_cube / 1e6:.1f} MB "
          f"per cube -> {per_cube * (len(tr_c) + len(va_c) + len(te_c)) / 1e9:.1f} GB on disk (memory-mapped); "
          f"one sample {SPAN * len(VARS) * P * P * 2 / 1e6:.2f} MB; train samples <= {len(tr_c) * NT * len(ISSUES)}",
          flush=True)
    if args.estimate:
        return
    out = os.path.join(OUT_BASE, args.tag) if args.tag else OUT_BASE
    ckdir, pdir = os.path.join(out, "neural_ckpt"), os.path.join(out, "neural_preds")
    for d in (CACHE, out, ckdir, pdir):
        os.makedirs(d, exist_ok=True)
    store = Store()
    t0 = time.time()
    split_ids, missing = {}, {}
    for nm, cl in (("train", tr_c), ("val", va_c), ("test", te_c)):
        ids, miss = [], []
        for b, y in cl:
            c = store.add(b, y)
            (miss.append(f"{b}:{y}") if c is None else ids.append(c))
        split_ids[nm], missing[nm] = ids, miss
        print(f"  {nm}: {len(ids)} cubes cached, {len(miss)} missing {miss[:8]}{'...' if len(miss) > 8 else ''}",
              flush=True)
    print(f"cache ready in {time.time() - t0:.0f} s", flush=True)
    if not split_ids["train"] or not split_ids["val"] or not split_ids["test"]:
        sys.exit("need at least one cube in each of train / val / test")
    if any(missing.values()) and not (args.allow_missing or args.smoke):
        sys.exit("cubes missing (see above): wait for the download or pass --allow-missing")
    if DEV == "cuda":
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        print(f"GPU free {free:.1f} GB of {torch.cuda.mem_get_info()[1] / 2 ** 30:.1f} GB", flush=True)
        if free < 7:
            print("  WARNING: another process holds GPU memory; peaks of ~5 GB may spill to shared memory "
                  "(WDDM) and run far slower", flush=True)
    tr_idx = store.index(split_ids["train"], drop_empty=True)
    va_idx = store.index(split_ids["val"], drop_empty=True)
    n_all = len(split_ids["train"]) * NT * len(ISSUES)
    print(f"train samples {len(tr_idx)} / {n_all} (non-empty), val samples {len(va_idx)}", flush=True)
    done = {}
    jp = os.path.join(out, "neural_runs.json")
    if args.skip_done and os.path.exists(jp):
        done = json.load(open(jp))
    commit = ef_commit()
    for seed in args.seeds:
        for name in args.models:
            run = f"{name}_s{seed}"
            if run in done:
                print(f"skip {run} (done)", flush=True); continue
            print(f"=== {run}", flush=True)
            model, norm, info = train_run(name, seed, args, store, tr_idx, va_idx, ckdir)
            ts = time.time()
            rows, preds, summ = score_test(model, norm, store, split_ids["test"], run, plan,
                                           8 if name == "earthformer" else 25, np.dtype(args.pred_dtype))
            pp = os.path.join(pdir, f"{run}.npz")
            np.savez_compressed(pp + ".tmp.npz", **preds); replace_retry(pp + ".tmp.npz", pp)
            info.update(test_summary=summ, test_minutes=(time.time() - ts) / 60,
                        pred_file=os.path.relpath(pp, ROOT), pred_mb=os.path.getsize(pp) / 1e6,
                        earthformer_commit=commit, earthformer_repo="https://github.com/amazon-science/earth-forecasting-transformer",
                        torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else None,
                        inputs=INPUTS, patch=P, tiles_per_window=NT, issues=ISSUES, leads=list(LEADS),
                        cubes={k: [f"{store.cubes[c][0]}:{store.cubes[c][1]}" for c in v] for k, v in split_ids.items()},
                        cubes_missing=missing,
                        schedule=f"linear warm-up {HP[name]['warmup']:.0%} then cosine to 1e-3 x lr over "
                                 f"{args.epochs} epochs x {info['steps_per_epoch']} steps",
                        loss="masked pinball, mean over scored pixels and P10/P50/P90",
                        output_param="P50 = last ffilled input NDTI + residual; P10/P90 = P50 -/+ softplus",
                        augmentation="random rot90 + horizontal flip (per batch)",
                        selection="best validation pinball epoch (val blocks, 2023)")
            write_outputs(out, run, rows, info)
            print(f"  [{run}] test {summ} | {info['train_minutes']:.1f} min train, peak GPU "
                  f"{info['peak_gpu_gb']:.2f} GB, {info['params'] / 1e6:.2f} M params, preds {info['pred_mb']:.1f} MB",
                  flush=True)
            if args.smoke:
                full_tr = 24000 * len(tr_idx) / max(1, len(split_ids["train"]) * NT * len(ISSUES))
                val_s = info["history"][-1]["val_sec"] / max(1, len(va_idx)) * 1500
                ep_min = (info["sec_per_step"] * full_tr / info["hyper"]["bs"] + val_s) / 60
                test_min = info["test_minutes"] / len(split_ids["test"]) * 64
                print(f"  [{run}] projection, full data (160 train cubes, {full_tr:.0f} samples at this "
                      f"non-empty rate): {ep_min:.1f} min/epoch, test scoring {test_min:.1f} min; "
                      + ", ".join(f"{e} epochs = {(e * ep_min + test_min) / 60:.1f} h" for e in (8, 12, 16)),
                      flush=True)
            del model
            torch.cuda.empty_cache()
    print("wrote", os.path.join(out, "neural_rows.csv"), flush=True)


if __name__ == "__main__":
    main()
