Title: Retrieval-Based Context-Aware Web QA Agent (Stateless)
Objective: Let users ask natural-language questions about any static webpage and get precise, grounded answers without retained memory.

Key Features:
	•	Loads arbitrary webpage content via WebBaseLoader.  ￼
	•	Splits text into overlapping chunks to preserve local context.
	•	Embeds chunks using LLaMA3 (via Ollama) and stores in Chroma vector store for similarity retrieval.
	•	Answers each question independently (stateless); filters out irrelevant queries with custom logic, returning a clear “out-of-scope” message when needed.

Tech Stack: Python, LangChain components (WebBaseLoader, RecursiveCharacterTextSplitter), Ollama’s LLaMA3, ChromaDB, vector similarity search.

Setup & Usage:
	1.	Clone repo and create .env with necessary Ollama endpoint / credentials.  ￼
	2.	Install dependencies from requirements.txt.  ￼
	3.	Run main.py, supply a URL, then ask questions via CLI.
	4.	Agent retrieves top-k similar chunks and answers grounded in page content; irrelevant questions are detected and responded to appropriately.

Impact: Eliminates manual keyword searches on static pages, enabling precise, natural-language information retrieval in one-shot queries.
