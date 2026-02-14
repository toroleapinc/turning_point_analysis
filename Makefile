.PHONY: install train evaluate visualize backtest test clean

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

visualize:
	python scripts/visualize.py

backtest:
	python scripts/backtest.py

test:
	pytest tests/ -v

clean:
	rm -rf saved_models/*.h5 saved_models/*.joblib data/*.csv
