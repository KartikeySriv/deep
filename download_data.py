from pathlib import Path
from urllib.request import urlretrieve


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FILE_PATH = DATA_DIR / "the-verdict.txt"

URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)


# ============================================================
# Download
# ============================================================

if FILE_PATH.exists():
    print(f"Dataset already exists: {FILE_PATH}")

else:
    print("Downloading The Verdict...")

    try:
        urlretrieve(URL, FILE_PATH)
        print(f"Dataset saved to: {FILE_PATH}")

    except Exception as error:
        print("Download failed.")
        print("Error:", error)