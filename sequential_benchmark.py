#!/usr/bin/env python3
"""
Sequential burned-area pipeline and benchmark
ARTI 503 Parallel Architecture and Programming - Project Stage 1
9FA01 - Group 1: Parallel Satellite Image Analysis for Burned Area Detection

For each satellite acquisition, ONE AT A TIME:
    1. load      - read the post-fire Sentinel-2 image and the reference mask
    2. preprocess- apply the validity mask, keep 12 bands, normalise, pad
    3. inference - run the U-Net on the full image (no tiling)
    4. postprocess - burned/unburned mask, burned area (km2), IoU and Dice
    5. save      - write the predicted mask to disk

Every stage is timed separately so the team can see which part takes longest.

Acknowledgement: the U-Net model (neural_net/unet.py, neural_net/unet_parts.py),
the 12-band input setup, the mask threshold (37/255) and the input
normalisation are taken from the authors' baseline code:
    Colomba, L. et al. (2022). burned-area-baseline. MIT License.
    https://github.com/lccol/burned-area-baseline
Dataset: Colomba, L. et al. (2022). Satellite Burned Area Dataset. Zenodo.
    https://zenodo.org/records/6597139
"""

import os

# --- Force a truly sequential run -----------------------------------------
# NumPy and PyTorch use several CPU threads by default. That would make the
# "sequential" baseline secretly parallel, so we limit everything to 1 thread
# BEFORE numpy / torch are imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import csv
import datetime
import glob
import json
import platform
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import tifffile
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

MASK_BURNED_THRESHOLD = 37    # baseline: grey values 37-255 = burned
N_BANDS = 12                  # baseline: all 12 Sentinel-2 L2A bands
STAGES = ["load", "preprocess", "inference", "postprocess", "save"]
DATE_RE = re.compile(r"sentinel2_(\d{4}-\d{2}-\d{2})\.tiff$")


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------
def find_acquisitions(data_dir):
    """Return a sorted list of acquisition folders that have a mask and a
    Sentinel-2 image. Sorting by name means every team member processes
    exactly the same acquisitions in the same order."""
    found = []
    for name in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            continue
        mask_path = os.path.join(folder, f"{name}_mask.tiff")
        s2_files = [f for f in glob.glob(os.path.join(folder, "sentinel2_*.tiff"))
                    if DATE_RE.search(os.path.basename(f))]
        if os.path.exists(mask_path) and s2_files:
            # The latest-dated Sentinel-2 image is the post-fire acquisition
            post = max(s2_files, key=lambda f: DATE_RE.search(os.path.basename(f)).group(1))
            found.append({"name": name, "image": post, "mask": mask_path})
    return found


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def load_stage(acq):
    image = tifffile.imread(acq["image"]).astype(np.float32)   # H x W x C
    mask = tifffile.imread(acq["mask"])
    if mask.ndim == 3:
        mask = mask[..., 0]
    return image, mask


def preprocess_stage(image):
    # Last channel is the validity mask: zero all bands where it is 0
    if image.shape[-1] > N_BANDS:
        invalid = image[..., -1] == 0
        image = image[..., :N_BANDS].copy()
        image[invalid] = 0
    else:
        image = image[..., :N_BANDS]

    # Reflectance to [0, 1] (handles both 0-1 floats and 0-10000 integers)
    if image.max() > 1.5:
        image = image / 10000.0
    image = np.clip(image, 0.0, 1.0)

    # Same normalisation as the baseline: (x - 0.5) / 0.5
    image = (image - 0.5) / 0.5

    # U-Net downsamples 4 times, so H and W must be multiples of 16
    h, w = image.shape[:2]
    pad_h, pad_w = (-h) % 16, (-w) % 16
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
    return tensor, (h, w)


def inference_stage(model, tensor):
    with torch.no_grad():
        logits = model(tensor)                     # 1 x 2 x H x W
    return logits


def postprocess_stage(logits, original_hw, ref_mask, pixel_size_m):
    h, w = original_hw
    pred = logits.argmax(dim=1)[0, :h, :w].numpy().astype(np.uint8)   # 1 = burned
    ref = (ref_mask[:h, :w] >= MASK_BURNED_THRESHOLD).astype(np.uint8)

    burned_pixels = int(pred.sum())
    area_km2 = burned_pixels * pixel_size_m ** 2 / 1e6

    inter = int(np.logical_and(pred, ref).sum())
    union = int(np.logical_or(pred, ref).sum())
    total = int(pred.sum() + ref.sum())
    iou = inter / union if union else 1.0
    dice = 2 * inter / total if total else 1.0
    return pred, {"burned_pixels": burned_pixels, "burned_area_km2": area_km2,
                  "iou": iou, "dice": dice}


def save_stage(pred, out_dir, name):
    tifffile.imwrite(os.path.join(out_dir, f"{name}_pred.tiff"), pred * 255)


def process_acquisition(model, acq, out_dir, pixel_size_m):
    """Run the full pipeline on one acquisition and time each stage."""
    t = {}
    s = time.perf_counter()
    image, ref_mask = load_stage(acq)
    t["load"] = time.perf_counter() - s

    s = time.perf_counter()
    tensor, hw = preprocess_stage(image)
    t["preprocess"] = time.perf_counter() - s

    s = time.perf_counter()
    logits = inference_stage(model, tensor)
    t["inference"] = time.perf_counter() - s

    s = time.perf_counter()
    pred, metrics = postprocess_stage(logits, hw, ref_mask, pixel_size_m)
    t["postprocess"] = time.perf_counter() - s

    s = time.perf_counter()
    save_stage(pred, out_dir, acq["name"])
    t["save"] = time.perf_counter() - s

    metrics.update({"height": hw[0], "width": hw[1]})
    return t, metrics


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(repo_dir, weights):
    sys.path.insert(0, os.path.abspath(repo_dir))
    from neural_net.unet import UNet          # authors' U-Net (MIT)
    torch.manual_seed(0)
    model = UNet(n_channels=N_BANDS, n_classes=2, act="relu")
    if weights:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    else:
        print("WARNING: no --weights given, using random weights. Timing is "
              "still valid (same computation), but IoU/Dice are meaningless.")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# System info (required by the Stage 1 handout)
# ---------------------------------------------------------------------------
def cpu_info():
    model = platform.processor() or "unknown"
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        break
        elif sys.platform == "darwin":
            model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        elif sys.platform.startswith("win"):
            out = subprocess.check_output(["wmic", "cpu", "get", "Name"], text=True)
            model = [l.strip() for l in out.splitlines() if l.strip()][1]
    except Exception:
        pass
    info = {"cpu_model": model, "logical_cores": os.cpu_count(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "torch": torch.__version__}
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        info["max_freq_mhz"] = round(freq.max) if freq and freq.max else None
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Sequential burned-area benchmark")
    p.add_argument("--data", required=True, help="folder containing the acquisition subfolders")
    p.add_argument("--repo", default="burned-area-baseline", help="path to the cloned baseline repo")
    p.add_argument("--weights", default=None, help="trained U-Net weights (.pt), optional")
    p.add_argument("--sizes", default="5,10,15,20", help="number of acquisitions to test")
    p.add_argument("--repeats", type=int, default=5, help="runs per input size")
    p.add_argument("--member", default="member", help="your name, used in the result files")
    p.add_argument("--pixel-size", type=float, default=10.0, help="pixel size in metres")
    p.add_argument("--out", default="results", help="output folder")
    args = p.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    acquisitions = find_acquisitions(args.data)
    if not acquisitions:
        sys.exit(f"No acquisitions found in {args.data}")
    if max(sizes) > len(acquisitions):
        print(f"Only {len(acquisitions)} acquisitions found; limiting sizes.")
        sizes = sorted({min(s, len(acquisitions)) for s in sizes})

    mask_dir = os.path.join(args.out, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    info = cpu_info()
    info.update({"member": args.member, "torch_threads": torch.get_num_threads(),
                 "date": datetime.datetime.now().isoformat(timespec="seconds")})
    with open(os.path.join(args.out, f"system_info_{args.member}.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))

    s = time.perf_counter()
    model = build_model(args.repo, args.weights)
    print(f"Model loaded in {time.perf_counter() - s:.2f} s (not included in run times)")

    # Warm-up on one acquisition so the first timed run is not slowed by
    # one-off start-up costs (memory allocation, library loading)
    process_acquisition(model, acquisitions[0], mask_dir, args.pixel_size)

    runs_path = os.path.join(args.out, f"runs_{args.member}.csv")
    acq_path = os.path.join(args.out, f"per_acquisition_{args.member}.csv")
    run_fields = ["member", "cpu_model", "n_acquisitions", "repeat", "total_s"] + [f"{st}_s" for st in STAGES]
    acq_fields = ["member", "n_acquisitions", "repeat", "acquisition", "height", "width",
                  "burned_pixels", "burned_area_km2", "iou", "dice"] + [f"{st}_s" for st in STAGES]

    totals = {}
    with open(runs_path, "w", newline="") as fr, open(acq_path, "w", newline="") as fa:
        wr, wa = csv.DictWriter(fr, run_fields), csv.DictWriter(fa, acq_fields)
        wr.writeheader(); wa.writeheader()
        for n in sizes:
            for r in range(1, args.repeats + 1):
                stage_sum = {st: 0.0 for st in STAGES}
                run_start = time.perf_counter()
                for acq in acquisitions[:n]:                # strictly one at a time
                    t, m = process_acquisition(model, acq, mask_dir, args.pixel_size)
                    for st in STAGES:
                        stage_sum[st] += t[st]
                    wa.writerow({"member": args.member, "n_acquisitions": n, "repeat": r,
                                 "acquisition": acq["name"], **m,
                                 **{f"{st}_s": round(t[st], 4) for st in STAGES}})
                total = time.perf_counter() - run_start
                totals.setdefault(n, []).append((total, stage_sum))
                wr.writerow({"member": args.member, "cpu_model": info["cpu_model"],
                             "n_acquisitions": n, "repeat": r, "total_s": round(total, 4),
                             **{f"{st}_s": round(stage_sum[st], 4) for st in STAGES}})
                fr.flush(); fa.flush()
                print(f"N={n:>2} run {r}/{args.repeats}: {total:.2f} s")

    # Summary: median total time and share of each stage
    print("\n=== Summary (median of repeats) ===")
    print(f"{'N':>3} {'total (s)':>10}  " + "  ".join(f"{st:>11}" for st in STAGES))
    for n, runs in totals.items():
        med_total = statistics.median(t for t, _ in runs)
        shares = []
        for st in STAGES:
            med = statistics.median(ss[st] for _, ss in runs)
            shares.append(f"{100 * med / med_total:10.1f}%")
        print(f"{n:>3} {med_total:>10.2f}  " + "  ".join(shares))
    print(f"\nResults saved to {runs_path} and {acq_path}")


if __name__ == "__main__":
    main()
