.PHONY: sync test lint demo eval download-demo

sync:
	uv sync --dev

lint:
	uv run ruff check .

test:
	uv run pytest -q

download-demo:
	uv run python scripts/download_demo_video.py

demo: download-demo
	uv run cv-traffic-counter \
		--source data/demo/street_corner.mp4 \
		--line 400,480,1000,480 \
		--zone 430,380,1050,380,1150,560,350,560 \
		--out data/demo/street_corner_annotated.mp4

eval: download-demo
	uv run python evals/run_eval.py
