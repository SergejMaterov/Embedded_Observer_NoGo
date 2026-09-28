.PHONY: install test checks clean

install:
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest

# regenerate the numbers reported in Appendix A of the paper
checks:
	python embedded_observer_checks.py | tee results/checks_output.txt

clean:
	rm -rf __pycache__ tests/__pycache__ .pytest_cache
