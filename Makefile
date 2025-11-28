.SILENT:

.PHONY: all
all:
	python -c "import glob, subprocess, sys; [subprocess.run([sys.executable, f]) for f in glob.glob('*.py')]"

.PHONY: format
format:
	ruff format *.py

.PHONY: check
check: format
	ruff check *.py
