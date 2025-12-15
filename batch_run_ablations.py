from pathlib import Path
import subprocess
import sys
import time

config_idx = int(sys.argv[1])
gpuid = sys.argv[2]

data_dir = Path("/iacl/pg25/sam/data/ECLARE_journal/ablation-data/LR/")
out_root = Path("/iacl/pg25/sam/data/ECLARE_journal/ablation-results/")
fpaths = sorted(data_dir.glob("*.nii*"))

# extra arguments based on the configuration

# Call the configurations:
# Baseline
# F (fov-aware resampling)
# E (ESPRESO)
# W (WDSR pixel shuffle modifications)
# FE (F and E)
# etc.
# FEW == ECLARE
configs = [
    ("--disable-fov-aware-resampling --interp-wdsr --relative-slice-thickness 5.2", "baseline"),
    ("--interp-wdsr --relative-slice-thickness 5.2", "F"),
    ("--disable-fov-aware-resampling --interp-wdsr", "E"),
    ("--disable-fov-aware-resampling --relative-slice-thickness 5.2", "W"),
    ("--interp-wdsr", "FE"),
    ("--relative-slice-thickness 5.2", "FW"),
    ("--disable-fov-aware-resampling", "EW"),
    ("", "eclare"),
]

config, name = configs[config_idx]
out_dir = out_root / name
out_dir.mkdir(parents=True, exist_ok=True)

for i, fpath in enumerate(fpaths):
    cmd = f"run-eclare --in-fpath {fpath} --out-dir {out_dir} --gpu-id {gpuid} {config}"

    st = time.time()
    print(f"Processing subject [{i}/{len(fpaths)}]", end=' ')
    subprocess.run(cmd, shell=True)
    en = time.time()
    mins = (en - st) / 60
    print(f"Elapsed: {mins:.2f} minutes")
