# Sequential Burned-Area Benchmark — ARTI 503 Stage 1

9FA01 – Group 1 · Parallel Satellite Image Analysis for Burned Area Detection

This script runs the burned-area pipeline **sequentially** (one acquisition at a time, one CPU thread) and times every stage. Every team member runs it on their own machine and sends back the result files.

## What it does

For each acquisition, in order:

| Stage | What happens |
| --- | --- |
| load | Read the post-fire Sentinel-2 image and the reference mask |
| preprocess | Apply the validity mask, keep the 12 bands, normalise, pad to a multiple of 16 |
| inference | Run the U-Net on the full image (no tiling) |
| postprocess | Make the burned/unburned mask, compute burned area (km²), IoU and Dice |
| save | Write the predicted mask to `results/masks/` |

The run is forced to a single thread (`torch.set_num_threads(1)` and `OMP_NUM_THREADS=1`), so the baseline is truly sequential.

## How to run (Google Colab or your laptop)

```bash
# 1. Get the baseline code (provides the U-Net model)
git clone https://github.com/lccol/burned-area-baseline.git

# 2. Get the data (part 1 is enough to start, 3.5 GB)
wget https://zenodo.org/records/6597139/files/Satellite_burned_area_dataset_part1.zip
unzip -q Satellite_burned_area_dataset_part1.zip -d data

# 3. Install packages
pip install -r requirements.txt

# 4. Run (change YourName)
python sequential_benchmark.py --data data/<folder with the EMSR... subfolders> --member YourName
```

`--data` must point to the folder whose subfolders are named like `EMSR207_04AVELAR_...`. Each subfolder holds `sentinel2_<date>.tiff` files and `<subfolder>_mask.tiff`.

In Colab, use a **CPU runtime** (Runtime → Change runtime type → CPU), and put `!` in front of each command.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--sizes` | `5,10,15,20` | Number of acquisitions per run (scalability) |
| `--repeats` | `5` | Runs per size; the summary reports the median |
| `--weights` | none | Trained U-Net weights (.pt). Without them timing is valid but IoU/Dice are not |
| `--member` | `member` | Your name, used in the file names |
| `--out` | `results` | Output folder |

Everyone must use the same `--sizes` and `--repeats` so results can be compared. Acquisitions are taken in alphabetical order, so everyone processes the same images.

## Output files (send these to Abrar, Member 8)

- `runs_<name>.csv` — total time and time per stage for each run
- `per_acquisition_<name>.csv` — time, image size, burned area, IoU and Dice per acquisition
- `system_info_<name>.json` — CPU model, cores, RAM, OS (needed for the CPU specs table)

The terminal also prints a summary showing which stage takes the largest share of the time.

## Notes

- **Weights:** the authors do not publish trained weights. Timing is the same with or without them, so benchmarking can start now. Train the model once on Colab before the final IoU/Dice results.
- **Pixel size:** burned area assumes 10 m pixels (`--pixel-size` to change).
- **Model load** time is printed but not included in run times. One warm-up run is done before timing starts.

## Acknowledgement

The U-Net model, 12-band input, mask threshold and normalisation come from the authors' baseline code: Colomba, L. et al. (2022). *burned-area-baseline*. MIT License. https://github.com/lccol/burned-area-baseline

Dataset: Colomba, L. et al. (2022). *Satellite Burned Area Dataset*. Zenodo. https://zenodo.org/records/6597139
