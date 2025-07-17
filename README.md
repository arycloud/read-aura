# Semantic Book Recommender

A web application that recommends books based on semantic search, categories, and emotional tones using OpenAI embeddings, LangChain, Chroma vector database, and Gradio for the user interface.

## Features
- **Semantic Search:** Find books by meaning, not just keywords.
- **Category & Emotion Filtering:** Filter recommendations by book category and emotional tone (Happy, Surprising, Angry, Suspenseful, Sad).
- **Interactive UI:** User-friendly web dashboard built with Gradio.

## Project Structure
- `gradio_dashboard.py` — Main application and UI logic
- `books_with_emotions.csv` — Book metadata with emotion scores
- `books_with_categories.csv`, `book_cleaned.csv` — Additional book data
- `tagged_description.txt` — Text data for semantic search
- `chroma_db/` — Persistent vector database
- `data-exploration.ipynb`, `sentiment-analysis.ipynb`, `vector-search.ipynb` — Notebooks for data analysis
- `.env` — Store your OpenAI API key (not tracked by git)

## Setup Instructions

### 1. Clone the Repository
```
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create and Activate a Virtual Environment (Recommended)
```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```
If `requirements.txt` is missing, install manually:
```
pip install pandas numpy python-dotenv gradio langchain chromadb openai
```

### 4. Add Your OpenAI API Key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

### 5. Prepare Data
Ensure the following files are present in the project root:
- `books_with_emotions.csv`
- `tagged_description.txt`

### 6. Run the Application
```
python gradio_dashboard.py
```
The Gradio dashboard will launch in your browser.

## Usage
- Enter a reading mood or topic in the search box.
- Select a category and emotional tone (optional).
- Click "Recommend Books" to see personalized recommendations.
- Click on a book to view details.

## Notes
- The first run may take longer as the vector database is built.
- Keep your `.env` file private and never commit it to GitHub.

## License
MIT License
