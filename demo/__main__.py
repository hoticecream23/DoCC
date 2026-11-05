import os
from pathlib import Path

from demo.server import serve

serve(int(os.environ.get("DOCC_DEMO_PORT", "8765")), Path(__file__).resolve().parent.parent)
