# College AI Chatbot

A local Streamlit chatbot that answers questions using the college PDFs in
`data/`. It retrieves relevant passages from a FAISS index and generates an
answer with Google's FLAN-T5 model. It does not use a hosted AI API, database,
or user account.

## Features

- Retrieval-augmented answers based on the provided college documents.
- Source PDF and page metadata stored with each indexed passage.
- Conversation history for the current Streamlit session, with a retry action
  when generation fails.
- Clear input validation and user-visible setup/model errors.
- CPU support, with CUDA used automatically when available.
- JSON document metadata and a FAISS binary index; no Python-pickle document
  store is loaded.

## Technology

- Python 3.10+
- Streamlit
- Hugging Face Transformers and Sentence Transformers
- LangChain PDF loading and text splitting
- FAISS CPU

## Project structure

```text
chatbot/
├── app.py                 # Streamlit interface
├── chatbot.py             # Retrieval, model loading, and answer generation
├── create_vector_db.py    # Build a safe FAISS index from the source PDFs
├── requirements.txt
├── data/                  # Source college PDFs
├── vectorstore/           # Generated index (ignored by Git)
└── tests/
    ├── test_app.py
    └── test_chatbot.py
```

## Prerequisites

- Python 3.10 or newer.
- Internet access the first time the embedding and generation models are
  downloaded from Hugging Face. Subsequent runs can use their local cache.
- Enough memory for `google/flan-t5-base`; inference runs on CPU if CUDA is not
  available.

## Install and run

From the project directory, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Build the local knowledge base from the PDFs, then start the app:

```powershell
python create_vector_db.py
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally
`http://localhost:8501`). Rebuild the index after changing or adding PDFs in
`data/`. Index generation downloads the sentence-embedding model if it is not
cached.

The generated `vectorstore/` directory is intentionally excluded from Git.
After cloning the project, run the index-building command before starting the
chatbot.

## Configuration

There are no API keys, environment variables, database credentials, or
`.env` file required. Both models run locally after their initial download.
The embedding model is `sentence-transformers/all-MiniLM-L6-v2`; the answer
generation model is `google/flan-t5-base`.

## Tests

Run the unit and Streamlit UI tests from the project directory:

```powershell
python -m unittest discover -s tests -v
```

Tests cover question validation, prompt grounding, retrieval and generation
behavior, error display, and retry handling. Model and network calls are mocked;
the test suite does not download models or contact external services.

## Limitations

- Answers depend on the contents and text quality of the supplied PDFs.
- Chat history is held only in the active Streamlit session and is not saved
  between sessions.
- Model downloads and response generation can take time, especially on CPU.
- There is no authentication, database, hosted API, or deployment configuration.

## Screenshots

To document the UI for a repository, capture the initial chat screen and a
sample conversation showing both a question and its answer. Do not include
private information in screenshots.

## Future improvements

- Add citations that link each generated answer to its retrieved PDF pages.
- Add OCR support for image-only PDFs.
- Add optional persistent conversation storage if the project later requires it.
