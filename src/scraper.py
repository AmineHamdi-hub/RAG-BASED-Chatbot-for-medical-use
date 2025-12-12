import requests
from bs4 import BeautifulSoup
import time
import random
from pathlib import Path

DOCS_DIR = Path("data/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive"
}

def scrape_medical_page(url: str, filename: str):
    # Retry loop (recommended for NCBI)
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            break
        except Exception as e:
            print(f"[Attempt {attempt+1}] Failed to fetch {url}: {e}")
            if attempt == 2:
                return None
            time.sleep(1 + random.random())
    
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts and styles
    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text(separator="\n")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    filepath = DOCS_DIR / filename
    filepath.write_text(text, encoding="utf-8")

    print(f"Saved {filepath}")
    return filepath


def scrape_multiple_pages(pages: dict):
    saved_files = []
    for fname, url in pages.items():
        path = scrape_medical_page(url, fname)
        if path:
            saved_files.append(path)
    return saved_files
