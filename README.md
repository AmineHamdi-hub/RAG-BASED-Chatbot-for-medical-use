# Medical Chatbot (RAG)

A minimal Retrieval-Augmented-Generation (RAG) example that scrapes medical pages, stores vector embeddings in Postgres (pgvector), and serves a simple Streamlit chat UI backed by the Groq LLM.

Key files:
- [app.py](app.py)
- [requirements.txt](requirements.txt)
- [.env.example](.env.example)
- [data/docs](data/docs)
- [src/llm_wrapper.py](src/llm_wrapper.py) — [`GroqLLM`](src/llm_wrapper.py)
- [src/scraper.py](src/scraper.py) — [`scrape_medical_page`](src/scraper.py), [`scrape_multiple_pages`](src/scraper.py)
- [src/rag.py](src/rag.py) — [`load_docs_from_folder`](src/rag.py), [`ingest_documents`](src/rag.py), [`store_embeddings`](src/rag.py), [`retrieve_similar`](src/rag.py), [`answer_question`](src/rag.py)
- [src/agent.py](src/agent.py) — [`MedicalRAGAgent`](src/agent.py)
- [src/pipeline.py](src/pipeline.py) — [`generate_medical_urls`](src/pipeline.py), [`scrape_urls`](src/pipeline.py), [`process_docs`](src/pipeline.py), [`extract_valid_urls`](src/pipeline.py), [`run_pipeline`](src/pipeline.py)

Prerequisites
- Python 3.10+
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension installed. Ensure the extension is enabled: `CREATE EXTENSION IF NOT EXISTS vector;`
- Groq API key

Setup
1. Create a Python virtual environment and install dependencies:
```bash
python -m venv myvenv
myvenv/Scripts/activate  # Windows
# or
source myvenv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# edit .env to set GROQ_API_KEY and POSTGRES_URI (example in .env.example)
```

3. Verify the database connection and that `POSTGRES_URI` points to a running Postgres with pgvector extension installed and reachable.

Usage

- Run the Streamlit app:
```bash
streamlit run app.py
```
This launches the web UI in [app.py](app.py), which uses the [`MedicalRAGAgent`](src/agent.py) and the Groq LLM wrapper [`GroqLLM`](src/llm_wrapper.py).

- Run the full pipeline (generate URLs, scrape, store embeddings):
```bash
python src/pipeline.py
```
Notes:
- The pipeline will call [`generate_medical_urls`](src/pipeline.py), then [`scrape_urls`](src/pipeline.py), then [`process_docs`](src/pipeline.py] which calls [`load_docs_from_folder`](src/rag.py), [`ingest_documents`](src/rag.py), [`store_embeddings`](src/rag.py) and [`build_graph`](src/rag.py) if present. Adjust as needed.
- The pipeline stores chunk embeddings in Postgres using [`store_embeddings`](src/rag.py). Ensure `POSTGRES_URI` is valid.

- Scrape pages manually:
```py
from src.scraper import scrape_multiple_pages
pages = {"page_1.txt": "https://example.com/article"}
scrape_multiple_pages(pages)
```
Scraped files are saved to [data/docs](data/docs).

- Query / Test a question via agent:
```py
from src.agent import MedicalRAGAgent
from src.llm_wrapper import GroqLLM

agent = MedicalRAGAgent(GroqLLM())
print(agent.answer("What are symptoms of influenza?"))
```

Design notes
- RAG: The `retrieve_similar` function in [src/rag.py](src/rag.py) uses `SentenceTransformer` to create embeddings and a Postgres table with a `vector(384)` column to store vectors.
- The chat UI in [app.py](app.py) uses Streamlit session state for the agent and history and delegates queries to [`MedicalRAGAgent`](src/agent.py).
- The scraper in [src/scraper.py](src/scraper.py) performs HTML cleaning and stores plain text files under [data/docs](data/docs).

Security & Legal
- This is an educational demo. The bot provides general information only and is not medical advice.
- Do not commit your `.env` or API keys. Use `.env` for local development and secure secrets in production.
- Ensure your Postgres instance is secured and accessible only to authorized systems.

Troubleshooting
- "Missing GROQ_API_KEY": Update `.env` based on [.env.example](.env.example) and restart the app.
- "Database connection fails": Confirm `POSTGRES_URI` is set and that `vector` extension is enabled:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
- "Pipeline import failures": Run `python src/pipeline.py` from the project root to ensure local module imports resolve.

Contributing
- Fixes, docs, and improvements welcome. Follow usual Python best practices and linting.

License
- MIT — see [LICENSE](LICENSE) for details.
