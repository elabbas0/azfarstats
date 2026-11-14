# azfarstats

Statistics engine for AzFar stadium recordings: per-player movement stats, estimated ball events
and a top-down replay with synced split-view video, from the two fixed corner cameras.

## Pipeline

Put the XVR files (`XVR_ch<N>_main_<start>_<end>.mp4`) in `data/raw/`, then:

| Step | Command | What it does |
|---|---|---|
| 1 | `python -m azfar.calibrate` | One-time per pitch: fisheye lens + camera pose from the painted lines (`config/calib_annotations.json`, median backgrounds in `data/bg/`). |
| 2 | `python -m azfar.detect --ch ch2` (and `ch3`) | Person and ball detection at 5 samples/s. Slowest step. |
| 3 | `python -m azfar.refine` | One-time per pitch: tightens the calibration using players seen by both cameras. |
| 4 | `python -m azfar.drift` | Re-checks each camera's aim every 10 minutes and corrects it. |
| 5 | `python -m azfar.embed` | Appearance embeddings (OSNet) for each player once per second. |
| 6 | `python -m azfar.fuse` | Merges both views on the pitch plane into tracklets; drops people off the pitch. |
| 7 | `python -m azfar.ballscan` then `python -m azfar.ball` | Motion blobs on every frame, then the single most plausible ball path from detector hits and blobs (triangulated when both cameras see it, height solved along the sight line otherwise). |
| 7b | `python -m azfar.window --write` | Finds the match window: skips the warm-up (several balls on the pitch at once) and stops where the first camera stops. |
| 8 | `python -m azfar.identify --start HH:MM:SS --end HH:MM:SS` | Assigns tracklets to a fixed roster of players. |
| 9 | `python -m azfar.stats` / `python -m azfar.events` | Minutes, distance, speeds, sprints / touches, passes, shots, goals, assists, saves, dribbles. |
| 10 | `python -m azfar.video` then `python -m azfar.export` | Split-view video segments and the self-contained replay page (`data/out/replay.html`). |

`python -m azfar.sheet` writes a contact sheet (a row of crops per player) for checking identities
and putting names to numbers. Names, team overrides and match windows go in `config/session.json`.

## What to trust

- Positions, distance and speed are measured; the two cameras agree to about 0.4 m.
- Player identity is by appearance and continuity. Players in distinctive kit are followed reliably;
  players in near-identical kit can be swapped with each other.
- Ball events are estimates from ball and player positions. Goals are the weakest: the ball is not
  visible inside the net, so a goal is inferred from a centre kick-off or from an on-target shot
  after which the ball disappears at the goal mouth.

## Requirements

Python 3.11+, ffmpeg, `torch`, `ultralytics`, `openvino`, `opencv-python-headless`, `scipy`.
Model weights are downloaded separately into `data/models/` (YOLO11s; OSNet x0.5 MSMT17 from
`kaiyangzhou/osnet` on Hugging Face). Note that Ultralytics is AGPL-3.0 licensed.
