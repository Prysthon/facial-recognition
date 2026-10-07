from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUTHORIZED_DIR = PROJECT_ROOT / "data" / "raw" / "authorized"
TEST_DIR = PROJECT_ROOT / "data" / "raw" / "test"
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "results"

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
DISTANCE_METRIC = "cosine"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}