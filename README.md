# Company Knowledge Assistant (Beginner RAG + LLMOps)

A local command-line Retrieval-Augmented Generation (RAG) project using Python, LangChain, Chroma, and Ollama. It answers questions from sample company policy text files and logs basic operational information locally.

## Requirements

- Python 3.10 or newer
- Ollama installed and running: https://ollama.com/
- Enough RAM for the local model; performance depends on your computer

## 1. Download local models

Open a terminal and run:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

- `llama3.2` generates answers.
- `nomic-embed-text` creates embeddings for documents and questions.

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Build the document index

Put `.txt` policy documents in the `data` folder. Run:

```bash
python ingest.py
```

This reads each text file, splits it into overlapping chunks, creates embeddings, and stores them in `chroma_db/`. Running ingestion again rebuilds the index from the current `.txt` files.

## 4. Start the chatbot

```bash
python chat.py
```

Try questions such as:
- How many casual leave days do employees receive?
- Who must approve training?
- What security rules apply when working remotely?
- What is the company's travel reimbursement limit?

Type `exit` to stop.

## 5. Run the evaluation set

```bash
python evaluate.py
```

The script tests three questions with known source documents and one question whose answer is absent. Review outputs manually; the basic checks are not a full evaluation framework.

## Project structure

```text
company_knowledge_assistant/
├── data/
│   ├── leave_policy.txt
│   ├── training_policy.txt
│   └── remote_work_policy.txt
├── chroma_db/                 # generated locally; not committed
├── logs/                      # generated locally; not committed
├── ingest.py                  # document loading, chunking, indexing
├── chat.py                    # retrieval, prompt, answer, logging
├── evaluate.py                # small test set
├── requirements.txt
├── .gitignore
└── README.md
```

## How the RAG workflow works

1. **Load:** `ingest.py` reads `.txt` documents.
2. **Chunk:** `RecursiveCharacterTextSplitter` divides text into overlapping chunks.
3. **Embed:** Ollama converts each chunk to a vector using `nomic-embed-text`.
4. **Store:** Chroma saves vectors and metadata locally.
5. **Retrieve:** The chatbot finds up to three relevant chunks for a question.
6. **Generate:** `llama3.2` receives the question and retrieved context.
7. **Cite sources:** The CLI prints source filenames from retrieved chunks.
8. **Log:** `chat.py` records timestamp, prompt version, question, source names, latency, and answer length in `logs/requests.jsonl`.
9. **Evaluate:** `evaluate.py` runs a small set of test questions.

## GitHub submission

From the project folder:

```bash
git init
git add .
git commit -m "Build company knowledge assistant RAG project"
```

Create an empty repository on GitHub, then follow GitHub's instructions to add the remote and push the code. Do not commit private company documents, local vector database files, logs, API keys, or other confidential information.

## Troubleshooting

- **Connection refused / Ollama unavailable:** Start the Ollama application or service and try `ollama list`.
- **Model not found:** Run `ollama pull llama3.2` and `ollama pull nomic-embed-text`.
- **Vector database missing:** Run `python ingest.py` before `python chat.py`.
- **Package import error:** Activate the correct virtual environment and run `pip install -r requirements.txt`.
- **Slow responses:** Local generation speed depends on RAM, CPU/GPU, and model size.

## Security and limitations

- This is a teaching example, not a production-ready company system.
- Use synthetic or approved sample documents for practice.
- Local inference does not automatically guarantee security; protect the computer and files.
- Logs include user questions. Review the logging code and your organization's privacy policy before using real company data.
- RAG can still produce mistakes. Check the answer against the source before relying on it.
- The example displays source filenames, not precise page-level citations.