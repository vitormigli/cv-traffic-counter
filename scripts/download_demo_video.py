"""Downloads the demo video used by the README and evals/run_eval.py.

Not committed to the repo (keeps it light) — this fetches it on demand.
Source: Mixkit Stock Video Free License (commercial/personal use permitted):
https://mixkit.co/free-stock-video/traffic-light-directing-traffic-4272/
"""

import urllib.request
from pathlib import Path

URL = "https://assets.mixkit.co/videos/4272/4272-720.mp4"
DEST = Path(__file__).resolve().parents[1] / "data" / "demo" / "street_corner.mp4"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already have {DEST}")
        return
    print(f"Downloading {URL} -> {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    print("Done.")


if __name__ == "__main__":
    main()
