from pathlib import Path

#from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
#load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Catalog cols

PHOT_DICT = {

                'jwst.nircam.F115W' : 1.154, 
                'jwst.nircam.F150W' : 1.501,
                'jwst.nircam.F200W' : 1.988,
                'jwst.nircam.F277W' : 2.776,
                'jwst.nircam.F356W' : 3.565,
                'jwst.nircam.F410M' : 4.083,
                'jwst.nircam.F444W' : 4.402,

                'jwst.miri.F770W' : 7.7,
                'jwst.miri.F1000W' : 10.0,
                'jwst.miri.F1500W' : 15.0,
                'jwst.miri.F2100W' : 21.0,
}


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
