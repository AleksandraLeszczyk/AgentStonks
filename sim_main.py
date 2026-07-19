"""SimLab entry point: `streamlit run sim_main.py`."""
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

from simlab.app import build_ui

build_ui()
