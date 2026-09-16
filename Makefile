install:
	python -m pip install -r requirements.txt

test:
	python -m pytest tests -q

run:
	python run_pipeline.py

demo:
	python run_demo.py

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('artifacts').glob('*.pkl')]"
