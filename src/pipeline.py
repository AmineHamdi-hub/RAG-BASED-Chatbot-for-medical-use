from scraper import scrape_multiple_pages
from rag import ingest_documents, store_embeddings, load_docs_from_folder
from llm_wrapper import GroqLLM
import re

llm = GroqLLM()

# --------------------------
# 1️⃣ Generate medical URLs with Groq
# --------------------------
def generate_medical_urls(prompt="Give me 100 medical article URLs from reputable sources. Format ONLY as plain URLs, one per line. Makes sure they are valid links."):
    response = llm(prompt)
    lines = [line.strip() for line in response.splitlines() if line.strip()]

    urls = extract_valid_urls(lines)
    return urls[:100]


# --------------------------
# 2️⃣ Scrape pages and save to docs/
# --------------------------
def scrape_urls(urls):
    pages = {f"page_{i}.txt": url for i, url in enumerate(urls)}
    saved_files = scrape_multiple_pages(pages)
    return saved_files

# --------------------------
# 3️⃣ Process docs: ingest, store embeddings, build graph
# --------------------------
def process_docs():
    docs = load_docs_from_folder()
    chunks = ingest_documents(docs)
    store_embeddings(chunks)
    print(f"Processed {len(docs)} docs into {len(chunks)} chunks.")



def extract_valid_urls(text_list):
    clean_urls = []
    url_pattern = r"(https?://[^\s\)\]]+)"
    
    for line in text_list:
        match = re.search(url_pattern, line)
        if match:
            clean_urls.append(match.group(1))
    
    return clean_urls

# --------------------------
# 4️⃣ Full pipeline
# --------------------------
def run_pipeline():
    print("Generating medical URLs...")
    urls = generate_medical_urls()
    print(f"Got {len(urls)} URLs")

    print("Scraping URLs...")
    scrape_urls(urls)

    print("Processing docs...")
    process_docs()
    print("Pipeline finished!")

# --------------------------
# Example query
# --------------------------
if __name__ == "__main__":
    run_pipeline()
