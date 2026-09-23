# 0003. No live camera/RTSP integration committed to this repo

## Status
Accepted

## Context
The original ask was "something with my home camera" — detect cars, people,
etc. `cv2.VideoCapture` (and Ultralytics' `model.track`) already accept a
file path, a webcam index, or an RTSP/HTTP URL identically — there's no
code difference between "demo video" and "my camera's stream," only a
different string passed to `--source`.

## Decision
The repo ships only file-based demo support: `scripts/download_demo_video.py`
fetches a public-domain street clip, and the README's one-command demo runs
against that file. Nothing in the repo references a real camera, IP, or
credential. `--source` accepting an RTSP URL is documented in
`cli.py`'s docstring as a capability, not demonstrated with a real one.

## Consequences
- A portfolio repo showing a live feed of someone's actual home would leak
  their address, routine, and property layout to anyone who finds the
  GitHub link — the opposite of what a portfolio piece should do. The
  public demo needs a video with no such cost, hence the licensed stock
  clip.
- The pipeline is still genuinely usable against a real camera locally,
  unmodified: `cv-traffic-counter --source "rtsp://user:pass@<ip>:554/..."`
  — that RTSP URL and any credentials belong in the operator's own shell
  history or a local, gitignored `.env`, never in a file this repo tracks
  or a commit message.
- This project's `--source` argument was deliberately kept generic (not
  hardcoded to file paths) specifically so that door stays open without
  requiring any code change later — see `cli.py`'s docstring.
