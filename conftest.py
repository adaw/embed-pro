import importlib, sys
from pathlib import Path

# embed-pro.py has a hyphen — can't be imported directly.
# Load it as "embed_pro" module so tests can `import embed_pro`.
spec = importlib.util.spec_from_file_location("embed_pro", Path(__file__).parent / "embed-pro.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["embed_pro"] = mod
spec.loader.exec_module(mod)
