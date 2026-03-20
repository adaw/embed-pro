.PHONY: dash serve test

dash:
	.venv/bin/python dashboard.py

serve:
	.venv/bin/python embed-pro.py

test:
	.venv/bin/python -m pytest test_embed_pro.py -v
