"""Curate one measured trial into results/<name>/ for the public repo.

    python scripts/publish_result.py sessions/tomato1 results/tomato1 --title "..." [--trial 1] [--max-mb 20]

Copies what a reader needs to check the result and what `kepler replay` needs to
reproduce it with no video and no API key:
  replay.json        frame times, frame size, camera track, measured pivot, parameters, published numbers
  sam/               the raw SAM 3.1 wire output the result came from (forward + backfill pass)
  measurement.json   the full trial record (technician verdict, laws, warnings)
  plot.png           angle, period vs amplitude, amplitude decay
  overlay.mp4        the measurement overlay, re-encoded to stay under --max-mb
Raw footage is not copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def shrink_video(src: Path, dst: Path, max_mb: float) -> None:
    """Re-encode until it fits: scale to 720 px wide, then raise CRF."""
    for crf in (26, 30, 34, 38):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-vf", "scale='min(720,iw)':-2",
                        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(dst)], check=True)
        if dst.stat().st_size <= max_mb * 1e6:
            return
    raise SystemExit(f"overlay still larger than {max_mb} MB at CRF 38")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--title", required=True)
    ap.add_argument("--max-mb", type=float, default=20.0)
    a = ap.parse_args()
    s, out, n = Path(a.session), Path(a.out), a.trial
    rp = s / f"trial{n:02d}.replay.json"
    if not rp.exists():
        raise SystemExit(f"{rp} missing: re-run `kepler measure` on this clip first")
    data = json.loads(rp.read_text(encoding="utf-8"))
    (out / "sam").mkdir(parents=True, exist_ok=True)
    for c in data["sam_caches"]:
        shutil.copy2(s / c["file"], out / "sam" / c["file"])
    shutil.copy2(rp, out / "replay.json")
    # the trial record names local files by absolute path: publish the repo-relative names instead
    rec = json.loads((s / f"trial{n:02d}.json").read_text(encoding="utf-8"))
    rec.update(clip="(raw footage not published)", plot="plot.png",
               overlay="overlay.mp4" if (s / f"trial{n:02d}.overlay.mp4").exists() else "")
    (out / "measurement.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    shutil.copy2(s / f"trial{n:02d}.png", out / "plot.png")
    ov = s / f"trial{n:02d}.overlay.mp4"
    if ov.exists():
        shrink_video(ov, out / "overlay.mp4", a.max_mb)
    (out / "title.txt").write_text(a.title + "\n", encoding="utf-8")
    sizes = {p.name: p.stat().st_size for p in out.rglob("*") if p.is_file()}
    print(f"published {s} trial {n} -> {out}: " + ", ".join(f"{k} {v / 1e6:.1f} MB" for k, v in sizes.items()))


if __name__ == "__main__":
    main()
