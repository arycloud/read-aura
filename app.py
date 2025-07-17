
# Standard library imports
import os
# Third-party imports
import pandas as pd
import numpy as np
from dotenv import load_dotenv
# LangChain and vector DB imports
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
# Gradio for UI
import gradio as gr

# Load environment variables from .env file (e.g., OpenAI API key)
load_dotenv()

# Load books data with emotion and category tags
books = pd.read_csv("books_with_emotions.csv")
# Add a column for large thumbnails (for better UI display)
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Vector DB setup with persistence
# If the Chroma DB exists, load it; otherwise, create it from tagged descriptions
persist_dir = "chroma_db"
if os.path.exists(persist_dir):
    db_books = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
else:
    raw_documents = TextLoader("tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=persist_dir)
    db_books.persist()



def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None,
                                      initial_top_k: int = 30, final_top_k: int = 12) -> pd.DataFrame:
    """
    Retrieve book recommendations using semantic search and filter by category and emotional tone.
    Args:
        query (str): User's search query.
        category (str): Book category to filter.
        tone (str): Emotional tone to filter.
        initial_top_k (int): Number of top results from vector search.
        final_top_k (int): Number of results to return after filtering.
    Returns:
        pd.DataFrame: Filtered and sorted book recommendations.
    """
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]

    book_recs = books[books["isbn13"].isin(books_list)]

    # Filter by category if specified
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # Sort by selected emotional tone
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy_x", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise_x", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger_x", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear_x", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness_x", ascending=False)

    return book_recs.head(final_top_k).reset_index(drop=True)


def generate_book_html(row, idx):
    """
    Generate HTML for displaying a book in the gallery.
    Args:
        row (pd.Series): Book data row.
        idx (int): Index of the book in the list.
    Returns:
        str: HTML string for the book card.
    """
    authors_split = row["authors"].split(";")
    if len(authors_split) == 2:
        authors_str = f"{authors_split[0]} and {authors_split[1]}"
    elif len(authors_split) > 2:
        authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
    else:
        authors_str = row["authors"]

    return f"""
    <div onclick=\"selectBook({idx})\" style='cursor:pointer;'>
        <img src='{row['large_thumbnail']}' style='width:100%; border-radius:8px;'>
        <div class='book-meta'><b>{row['title']}</b><br><i>{authors_str}</i></div>
    </div>
    """


def recommend_books(query: str, category: str, tone: str):
    """
    Get book recommendations and generate HTML for the gallery.
    Args:
        query (str): User's search query.
        category (str): Book category.
        tone (str): Emotional tone.
    Returns:
        tuple: (HTML string, list of book dicts)
    """
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    html_items = [generate_book_html(row, idx) for idx, row in recommendations.iterrows()]
    return "".join(html_items), recommendations.to_dict("records")


def display_book_details(book):
    """
    Generate detailed HTML for a selected book.
    Args:
        book (dict): Book data.
    Returns:
        str: HTML string for book details.
    """
    if not book:
        return ""

    authors = book["authors"].replace(";", ", ")
    return f"""
    <div style='padding:1rem;'>
        <img src='{book['large_thumbnail']}' style='max-width:200px; float:left; margin-right:1rem;'>
        <h2>{book['title']}</h2>
        <h4>by {authors}</h4>
        <p>{book['description']}</p>
    </div>
    """


# UI Layout and Gradio App
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gr-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        margin-top: 0.5rem;
    }
    .gr-subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .book-meta {
        text-align: center;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
""") as dashboard:
    # App title and subtitle
    gr.Markdown('<div class="gr-title">📚 Semantic Book Recommender</div>')
    gr.Markdown('<div class="gr-subtitle">Discover books tailored to your interests and emotions</div>')

    # Layout: left column for input, right for results
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            user_query = gr.Textbox(
                label="🔍 What are you in the mood to read?",
                placeholder="e.g., A journey of self-discovery through nature",
                lines=3
            )
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="📖 Category",
                value="All",
                interactive=True
            )
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="🎭 Emotional Tone",
                value="All",
                interactive=True
            )
            submit_button = gr.Button("✨ Recommend Books", variant="primary")

        with gr.Column(scale=2, min_width=500):
            gallery_html = gr.HTML()
            book_detail = gr.HTML()
            book_data_state = gr.State()

    def handle_recommend(query, cat, tone):
        """
        Handle the recommend button click: get recommendations and update UI.
        """
        gallery, books_data = recommend_books(query, cat, tone)
        grid = f"""
        <script>
        window.selectBook = function(idx) {{
            const input = document.querySelector('input[name=\"__selected_book_index\"]');
            if(input) input.value = idx;
            input.dispatchEvent(new Event('input'));
        }}
        </script>
        <div style='display:grid; grid-template-columns: repeat(4, 1fr); gap: 16px;'>
            {gallery}
        </div>
        """
        return grid, books_data, ""

    def handle_click(index, books):
        """
        Handle book selection from the gallery.
        """
        if books and 0 <= index < len(books):
            return display_book_details(books[index])
        return ""

    selected_index = gr.Number(visible=False, label="__selected_book_index")

    # Connect UI events to functions
    submit_button.click(
        fn=handle_recommend,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=[gallery_html, book_data_state, book_detail]
    )

    selected_index.change(
        fn=handle_click,
        inputs=[selected_index, book_data_state],
        outputs=book_detail
    )

# Run the Gradio app
if __name__ == "__main__":
    dashboard.launch()
