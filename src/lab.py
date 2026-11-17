"""One command for the lab PC: start, or resume, stage 6.2 training.

    lab.bat                    (Windows: pulls, sets up .venv once, then runs this)
    python -m src.lab          start or resume, whichever applies
    python -m src.lab --status how far the run is, without starting anything
    python -m src.lab --backup E:\\capstone   also copy checkpoints + metrics there on exit

Pausing: press Ctrl+C at any time. Every finished epoch is already saved; the
epoch in progress is dropped. Running the command again continues from the
last saved epoch with the same learning-rate schedule.

What it does, in order:
1. Finds the dataset: data/Task07_Pancreas/, else extracts data/Task07_Pancreas.tar
   (just drop the tar into data/), else downloads the tar itself (resumable).
2. Refuses to train on CPU by accident (a CPU-only torch wheel is the usual cause).
3. Runs src.train_seg with the project's fixed split, --resume, and a time
   budget: the first session times one epoch and fixes the epoch count to fit
   `--hours` of training in total (run.json). Later sessions just continue.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

from .config import CHECKPOINT_DIR, DATA_DIR, MSD_DIR, REPO_ROOT, RESULTS_DIR

SPLIT = REPO_ROOT / "results" / "review1_local" / "full281" / "split_281cases.json"
OUT = RESULTS_DIR / "final"
CKPT = CHECKPOINT_DIR / "stage62"
DEFAULT_HOURS = 11.0       # ~6.5 h on day 1 + ~4.5 h on day 2 before the 14:00 inference


MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"
MSD_TAR_BYTES = 12_289_971_712   # the official tar; a smaller file is an unfinished copy


def download_msd(dest: Path, url: str = MSD_URL, chunk: int = 8 << 20) -> Path:
    """Resumable download to `dest`. Bytes land in `<dest>.part` and are renamed
    only when complete, so a half-downloaded file is never mistaken for the tar.
    Ctrl+C keeps the .part; the next run continues from where it stopped."""
    import time
    import urllib.request

    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r:
        if have and r.status != 206:              # server ignored Range: start over
            have = 0
        total = have + int(r.headers.get("Content-Length", 0))
        print(f"[lab] downloading MSD Task07 ({total / 1e9:.1f} GB)"
              + (f", resuming at {have / 1e9:.1f} GB" if have else "") + " - Ctrl+C keeps progress")
        t0, last = time.time(), -1
        with open(part, "ab" if have else "wb") as f:
            done = have
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                pct = int(100 * done / total) if total else 0
                if pct != last and pct % 2 == 0:
                    rate = (done - have) / max(time.time() - t0, 1e-6)
                    eta = (total - done) / rate / 60 if rate else float("nan")
                    print(f"[lab]   {pct:3d}%  {done / 1e9:5.1f} GB  {rate / 1e6:5.1f} MB/s  ~{eta:.0f} min left")
                    last = pct
    part.replace(dest)
    return dest


def find_dataset(data_dir: Path | None = None, msd_dir: Path | None = None,
                 download: bool = True) -> Path:
    """Return the MSD folder. In order of preference: an extracted folder, a
    tar in data/ (extracted once), or - with `download` - fetch the tar."""
    data_dir = Path(data_dir or DATA_DIR)
    msd_dir = Path(msd_dir or data_dir / MSD_DIR.name)
    if (msd_dir / "dataset.json").exists():
        return msd_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    tars = sorted(data_dir.glob("Task07_Pancreas*.tar"))
    if not tars:
        if not download:
            raise SystemExit(
                f"[lab] No dataset. Put Task07_Pancreas.tar (or the extracted Task07_Pancreas "
                f"folder) into {data_dir} and run again.")
        tars = [download_msd(data_dir / "Task07_Pancreas.tar")]
    size = tars[0].stat().st_size
    if MSD_TAR_BYTES and size != MSD_TAR_BYTES:
        raise SystemExit(
            f"[lab] {tars[0]} is {size / 1e9:.2f} GB but the MSD tar is {MSD_TAR_BYTES / 1e9:.2f} GB - "
            "an unfinished copy. Copy it again, or delete it and run with no tar to download it.")
    print(f"[lab] extracting {tars[0].name} into {data_dir} (one time, ~10 min) ...")
    with tarfile.open(tars[0]) as tf:
        members = [m for m in tf.getmembers() if not Path(m.name).name.startswith("._")]
        try:
            tf.extractall(data_dir, members=members, filter="data")
        except TypeError:                           # Python < 3.10.12 has no filter=
            tf.extractall(data_dir, members=members)
    if not (msd_dir / "dataset.json").exists():
        raise SystemExit(f"[lab] extracted, but {msd_dir / 'dataset.json'} is missing - "
                         "is this the MSD Task07 tar?")
    print("[lab] dataset ready")
    return msd_dir


def status(ckpt_dir: Path | None = None, out: Path | None = None) -> str:
    ckpt_dir, out = Path(ckpt_dir or CKPT), Path(out or OUT)
    run = ckpt_dir / "run.json"
    metrics = out / "metrics_stage62.csv"
    if not (ckpt_dir / "last.pt").exists():
        return "[lab] no training saved yet - the next run starts from epoch 0"
    planned = json.loads(run.read_text())["epochs"] if run.exists() else "?"
    lines = [f"[lab] saved run in {ckpt_dir}"]
    if metrics.exists():
        import pandas as pd
        df = pd.read_csv(metrics)
        done = int(df["epoch"].max()) + 1
        best = df["dice_mean"].max()
        pace = df.get("epoch_seconds", df["seconds"]).tail(5).mean()
        left = (planned - done) * pace / 3600 if isinstance(planned, int) else float("nan")
        lines.append(f"[lab] {done}/{planned} epochs done | best mean Dice {best:.4f} | "
                     f"~{left:.1f} h of training left")
        last = df.iloc[-1]
        lines.append(f"[lab] last epoch: pancreas Dice {last['dice_pancreas']} "
                     f"tumor Dice {last['dice_tumor']}")
    return "\n".join(lines)


def backup(dest, ckpt_dir: Path | None = None, out: Path | None = None) -> None:
    ckpt_dir, out = Path(ckpt_dir or CKPT), Path(out or OUT)
    dest = Path(dest)
    if not dest.parent.exists() and not dest.exists():
        print(f"[lab] backup skipped: {dest} is not reachable (USB drive plugged in?)")
        return
    (dest / "checkpoints" / "stage62").mkdir(parents=True, exist_ok=True)
    for f in ("last.pt", "best.pt", "run.json"):
        if (ckpt_dir / f).exists():
            shutil.copy2(ckpt_dir / f, dest / "checkpoints" / "stage62" / f)
    if out.exists():
        shutil.copytree(out, dest / "results_final", dirs_exist_ok=True)
    print(f"[lab] backed up checkpoints and results to {dest}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Lab PC: start or resume stage 6.2 training")
    p.add_argument("--status", action="store_true", help="show progress and exit")
    p.add_argument("--hours", type=float, default=DEFAULT_HOURS,
                   help="total training budget, used only by the very first session")
    p.add_argument("--backup", default="", help="folder (e.g. on a USB drive) to copy results to on exit")
    p.add_argument("--no-download", action="store_true",
                   help="fail instead of downloading the 12.3 GB tar when data/ is empty")
    p.add_argument("--allow-cpu", action="store_true", help="for testing the launcher only")
    p.add_argument("--fresh", action="store_true",
                   help="discard the saved run and start again from epoch 0")
    args, passthrough = p.parse_known_args(argv)

    if args.status:
        print(status())
        return

    try:
        data = find_dataset(download=not args.no_download)
    except KeyboardInterrupt:
        print("\n[lab] download paused - run lab.bat again to continue from where it stopped.")
        return

    import torch
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("[lab] PyTorch sees no GPU. Install the CUDA build:\n"
                         "  .venv\\Scripts\\python -m pip install torch --force-reinstall "
                         "--index-url https://download.pytorch.org/whl/cu124")
    if torch.cuda.is_available():
        print(f"[lab] GPU: {torch.cuda.get_device_name(0)}")

    print(status())
    if args.fresh:
        print("[lab] --fresh: the saved run will be overwritten")
    else:
        print("[lab] Ctrl+C pauses safely at any time; run this again to resume.")

    from .train_seg import main as train_seg
    cmd = ["--data-root", str(data), "--split", str(SPLIT), "--out", str(OUT),
           "--ckpt-root", str(CKPT.parent), "--hours", str(args.hours)]
    if not args.fresh:
        cmd.append("--resume")
    try:
        train_seg(cmd + passthrough)
    except KeyboardInterrupt:          # Ctrl+C during setup (data caching) - nothing lost
        print("\n[lab] stopped before the next epoch was saved; the saved run is unchanged.")
    finally:
        if args.backup:
            backup(args.backup)
    print(status())
    if (CKPT / "best.pt").exists():
        print(f"[lab] best model so far: {CKPT / 'best.pt'}")


if __name__ == "__main__":
    sys.exit(main())
