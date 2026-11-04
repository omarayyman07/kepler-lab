"""The lab's hands and eyes.

Releasers set up the experiment the agent asked for:
  * ManualReleaser: the agent tells you the angle; you release by hand.
    The amplitude is *measured* from video anyway, so this is scientifically
    sound; it only costs autonomy.
  * SerialReleaser: ESP32/Arduino with a servo arm and an electromagnet
    (firmware/kepler_release). The arm carries a steel bob to the angle
    from behind the swing plane, the magnet lets go.

Recorders capture the run:
  * CameraRecorder: any OpenCV source (USB webcam, or a phone used as a
    webcam). Writes per-frame wall-clock timestamps beside the clip, because
    webcams do not deliver a steady frame rate.
  * DropFolderRecorder: waits for a new video file to appear in a folder
    (record on the phone in slow motion, sync or copy it over).
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

import numpy as np

from . import status

VIDEO_EXT = {".mp4", ".mov", ".m4v"}


# --------------------------------------------------------------------------
# Releasers
# --------------------------------------------------------------------------

BAR = "─" * 64


class ManualReleaser:
    """A person sets up each trial; the experiment says what to do (`manual_instruction`),
    the recorder says how recording starts (`start_hint`). `auto` skips the Enter
    prompt (rehearsals), and prints everything else exactly as a real session would."""

    def __init__(self, instruction=None, start_hint: str = "", auto: bool = False):
        self.instruction = instruction or (lambda v: f"Set up the trial for {v}.")
        self.start_hint = start_hint
        self.auto = auto

    def setup(self, value: float, seconds: float | None = None) -> None:
        steps = [self.instruction(value)]
        if self.start_hint:
            steps.append(self.start_hint)
        steps.append("Press Enter in the terminal (or 'Ready – release' in the Lab Console), then let go when "
                     "you see RELEASE: open your fingers, do not push.")
        print(f"\n{BAR}\n  YOUR TURN\n{BAR}")
        for k, s in enumerate(steps, 1):
            print(f"  {k}. {s}")
        if seconds:
            print(f"     After that, keep recording for {seconds:.0f} s; a countdown will tell you when to stop.")
        token = secrets.token_hex(8)
        go = status.go_path()
        if go is not None:
            go.unlink(missing_ok=True)       # a click from an earlier prompt never counts
        status.write("your_turn", steps=steps, seconds=seconds, go_token=token)
        if self.auto:
            print(">>> (rehearsal: Enter pressed automatically)")
        else:
            how = status.wait_for_go(">>> Press Enter (or 'Ready – release' in the Lab Console) when you are "
                                     "holding the bob and recording ... ", token)
            # leave YOUR TURN at once, so a second click cannot land on this prompt
            status.write("released", message=f"Ready pressed ({how}). Let go now.")

    def release(self) -> None:
        print(f"\n{'#' * 24}\n####  RELEASE NOW  ####\n{'#' * 24}")


class SerialReleaser:
    """Protocol: 'ARM <deg>' -> 'READY'; 'DROP' -> 'DROPPED'; 'HOME' -> 'READY'."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 20.0):
        import serial  # pyserial

        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)  # boards reset when the port opens
        self.ser.reset_input_buffer()

    def _cmd(self, line: str, expect: str) -> None:
        self.ser.write((line + "\n").encode())
        deadline = time.time() + 30
        while time.time() < deadline:
            reply = self.ser.readline().decode(errors="replace").strip()
            if reply == expect:
                return
            if reply.startswith("ERR"):
                raise RuntimeError(f"Releaser: {reply}")
        raise TimeoutError(f"Releaser did not answer {expect} to {line}")

    def setup(self, angle_deg: float, seconds: float | None = None) -> None:
        self._cmd(f"ARM {angle_deg:.1f}", "READY")
        time.sleep(1.5)  # let the bob settle against the magnet

    def release(self) -> None:
        self._cmd("DROP", "DROPPED")


# --------------------------------------------------------------------------
# Recorders
# --------------------------------------------------------------------------

class CameraRecorder:
    start_hint = "The webcam starts recording by itself when you press Enter; check the bob is in view."

    def __init__(self, source: int | str = 0, width: int = 1280, height: int = 720, fps: int = 60):
        self.source, self.size, self.fps = source, (width, height), fps

    def record(self, out: str | Path, seconds: float, on_start=None) -> Path:
        import cv2

        out = Path(out)
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Camera {self.source} gave no frames")
        h, w = frame.shape[:2]
        status.write("recording", until=time.time() + seconds, seconds=seconds)
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        stamps = []
        t0 = time.perf_counter()
        started = False
        while time.perf_counter() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            stamps.append(time.perf_counter() - t0)
            vw.write(frame)
            if not started and on_start and stamps[-1] > 0.5:
                on_start()
                started = True
        cap.release()
        vw.release()
        out.with_suffix(".timestamps.json").write_text(json.dumps(stamps))
        return out


class DropFolderRecorder:
    """A phone records; you copy each clip into a folder. We count down the recording,
    then wait for a new video file to appear and finish copying."""

    start_hint = ("Start recording on the phone now (on its stand, normal video, not slow-motion; "
                  "the whole swing in frame).")

    def __init__(self, folder: str | Path):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)

    def _videos(self) -> set[Path]:
        return {p for p in self.folder.iterdir() if p.suffix.lower() in VIDEO_EXT}

    def _countdown(self, seconds: float) -> None:
        status.write("recording", until=time.time() + seconds, seconds=seconds)
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            print(f"\r   Keep recording, hands off: {left:4.0f} s left ", end="", flush=True)
            time.sleep(min(1.0, left))
        print(f"\r   Keep recording, hands off:    0 s left ")

    def record(self, out: str | Path, seconds: float, on_start=None) -> Path:
        before = self._videos()
        if on_start:
            on_start()
        self._countdown(seconds)
        print(f"\n  4. STOP recording now.\n  5. Copy the video (original quality) into:\n       {self.folder}\n"
              "     Waiting for it to appear ...", flush=True)
        status.write("copy", steps=["STOP recording now.",
                                    f"Copy the video (original quality) into {self.folder}"])
        while True:
            new = sorted(self._videos() - before, key=lambda p: p.stat().st_mtime)
            if new:
                p = new[-1]
                size = -1
                while size != p.stat().st_size:  # wait for the copy to finish
                    size = p.stat().st_size
                    time.sleep(1.0)
                print(f"   Got {p.name} ({size / 1e6:.1f} MB). Measuring ...")
                status.write("measuring", clip=p.name)
                return p
            time.sleep(1.0)


def load_sidecar_timestamps(clip: str | Path, n_frames: int) -> np.ndarray | None:
    side = Path(clip).with_suffix(".timestamps.json")
    if not side.exists():
        return None
    ts = np.array(json.loads(side.read_text()), float)
    return ts - ts[0] if ts.size == n_frames else None
