# simple shortcuts so I do not have to remember the long commands.
# usage: "make setup", "make test", "make demo"

.PHONY: setup test demo

setup:
	pip install -r requirements-dev.txt

test:
	pytest -q

demo:
	python model.py
	python fairness_audit.py
	python llm_reliability.py
