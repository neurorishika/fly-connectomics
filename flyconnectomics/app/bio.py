import os
import json
import sqlite3
import atexit
from datetime import datetime, timedelta

import torch
import numpy as np
import requests
from flask import Flask, render_template, request, Response, stream_with_context, session, redirect, url_for, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from apscheduler.schedulers.background import BackgroundScheduler

# --- Configuration ---
DATABASE = 'articles.db'
# Using os.urandom for a secure, randomly generated secret key on each startup.
# For production, set this from an environment variable.
SECRET_KEY = os.urandom(24)
# Determine if CUDA is available for GPU usage
DEVICE = 0 if torch.cuda.is_available() else -1

# --- App Initialization ---
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.secret_key = SECRET_KEY

# --- Model Loading ---
# Load models once at startup to avoid reloading on each request
print("Loading Hugging Face models... This may take a moment.")
MODELS = {
    'qa': pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        device=DEVICE
    ),
    'summarizer': pipeline(
        "summarization",
        model="Falconsai/text_summarization",
        device=DEVICE,
        clean_up_tokenization_spaces=True
    ),
    'embedding': SentenceTransformer('all-MiniLM-L6-v2')
}
print("Models loaded successfully.")

# --- Database Management ---
def get_db_connection():
    """Create a database connection with Row factory for dict-like access."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database schema."""
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                doi TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                category TEXT,
                pub_date TEXT
            );
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS votes (
                doi TEXT PRIMARY KEY,
                vote_type TEXT CHECK(vote_type IN ('upvote', 'downvote'))
            );
        ''')
        conn.commit()
        print("Database initialized.")

# --- Background Scheduler for Data Cleanup ---
def cleanup_old_data():
    """Deletes articles and corresponding votes older than 180 days."""
    cutoff_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM articles WHERE pub_date < ?', (cutoff_date,))
        # Also clean up votes for deleted articles
        cursor.execute('DELETE FROM votes WHERE doi NOT IN (SELECT doi FROM articles)')
        conn.commit()
        print(f"Cleanup complete: Articles older than {cutoff_date} removed.")

# Initialize and start the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=cleanup_old_data, trigger="interval", days=1)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# --- Core Logic Functions ---
def fetch_biorxiv_data(start_date, end_date):
    """Fetches article data from the bioRxiv API for a given date range."""
    base_url = f'https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}'
    cursor = 0
    batch_size = 100
    all_articles = []

    while True:
        try:
            response = requests.get(f'{base_url}/{cursor}')
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: Failed to fetch data. {e}")
            return None, f"Error fetching data from bioRxiv API: {e}"

        if not data.get('collection'):
            break # No more articles

        all_articles.extend(data['collection'])
        cursor += batch_size

    return all_articles, None

def add_articles_to_db(articles):
    """Adds a list of articles to the database, ignoring duplicates."""
    if not articles:
        return
    with get_db_connection() as conn:
        cursor = conn.cursor()
        article_data = [
            (
                article['doi'],
                article['title'],
                article['authors'],
                article['abstract'],
                article['category'],
                article['date'] # Use 'date' field from API as pub_date
            ) for article in articles
        ]
        # Use INSERT OR IGNORE to skip articles with a DOI that's already in the table
        cursor.executemany('''
            INSERT OR IGNORE INTO articles (doi, title, authors, abstract, category, pub_date)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', article_data)
        conn.commit()

def get_articles_from_db(start_date, end_date, categories=None):
    """Retrieves articles from the database within a date range and by category."""
    with get_db_connection() as conn:
        query = 'SELECT * FROM articles WHERE pub_date BETWEEN ? AND ?'
        params = [start_date, end_date]

        if categories:
            # Build a dynamic clause for categories using LIKE
            category_clauses = ' OR '.join(['category LIKE ?'] * len(categories))
            query += f' AND ({category_clauses})'
            params.extend([f'%{cat.strip()}%' for cat in categories])

        return conn.execute(query, tuple(params)).fetchall()

def rank_articles(articles, keywords, negative_keywords):
    """Ranks articles based on semantic similarity to keywords."""
    if not articles or not keywords:
        return articles, [1.0] * len(articles)

    abstracts = [article['abstract'] for article in articles]
    abstract_embeddings = MODELS['embedding'].encode(abstracts, convert_to_tensor=True)

    # Create a weighted keyword embedding
    weighted_embeddings = None
    for i, keyword in enumerate(keywords):
        # Give more weight to earlier keywords
        embedding = MODELS['embedding'].encode(keyword, convert_to_tensor=True) * (len(keywords) - i)
        weighted_embeddings = embedding if weighted_embeddings is None else weighted_embeddings + embedding

    # Subtract negative keyword embeddings
    for neg_keyword in negative_keywords:
        neg_embedding = MODELS['embedding'].encode(neg_keyword, convert_to_tensor=True)
        weighted_embeddings -= neg_embedding

    if weighted_embeddings is None:
        return articles, [1.0] * len(articles)

    # Normalize the final query embedding
    weighted_embeddings /= torch.norm(weighted_embeddings)

    # Calculate cosine similarity
    cosine_scores = util.cos_sim(weighted_embeddings, abstract_embeddings)[0].cpu()

    # Sort articles by score
    sorted_indices = np.argsort(-cosine_scores)
    sorted_articles = [articles[i] for i in sorted_indices]
    sorted_scores = [cosine_scores[i].item() for i in sorted_indices]

    return sorted_articles, sorted_scores

def generate_insight_and_summary(abstract):
    """Generates a structured insight and a summary for a given abstract."""
    insight_questions = [
        "What is the main objective of this study?",
        "What is the main result of this study?",
        "What is the significance of this study?",
    ]
    insights = []
    for q in insight_questions:
        try:
            result = MODELS['qa'](question=q, context=abstract)
            insights.append(result['answer'].capitalize())
        except Exception:
            insights.append("N/A")

    # Generate summary
    try:
        summary_text = MODELS['summarizer'](abstract, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    except Exception:
        summary_text = "Summary could not be generated."

    insight_str = f"Objective: {insights[0]} | Results: {insights[1]} | Significance: {insights[2]}"
    return summary_text, insight_str

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    """Renders the main page."""
    # Load last used values from session, defaulting to empty lists/string
    return render_template(
        'index.html',
        keywords=session.get('keywords', []),
        negative_keywords=session.get('negative_keywords', []),
        categories=session.get('categories', []),
        period_days=session.get('period_days', '30')
    )

@app.route('/generate', methods=['GET'])
def generate():
    """Endpoint to generate and stream summaries."""
    def stream_events():
        # --- 1. Get and Validate Parameters ---
        keywords = request.args.getlist('keywords')
        negative_keywords = request.args.getlist('negative_keywords')
        categories = request.args.getlist('categories')
        try:
            period_days = int(request.args.get('period_days', 30))
        except (ValueError, TypeError):
            period_days = 30

        # Save parameters to session
        session['keywords'] = keywords
        session['negative_keywords'] = negative_keywords
        session['categories'] = categories
        session['period_days'] = period_days

        if not keywords:
            yield f"data: {json.dumps({'error': 'Please provide at least one keyword.'})}\n\n"
            return
        
        yield f"data: {json.dumps({'message': 'Starting process...', 'progress': 0})}\n\n"

        # --- 2. Fetch and Cache Data ---
        end_date = datetime.today()
        start_date = end_date - timedelta(days=period_days)
        start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        
        yield f"data: {json.dumps({'message': f'Fetching new articles from bioRxiv for {start_str} to {end_str}...', 'progress': 5})}\n\n"
        articles_from_api, error = fetch_biorxiv_data(start_str, end_str)

        if error:
            yield f"data: {json.dumps({'error': error})}\n\n"
            return
        
        add_articles_to_db(articles_from_api)
        yield f"data: {json.dumps({'message': f'Found and cached {len(articles_from_api)} new articles. Retrieving relevant articles from database...', 'progress': 20})}\n\n"

        # --- 3. Retrieve and Filter Articles from DB ---
        db_articles = get_articles_from_db(start_str, end_str, categories)
        if not db_articles:
            yield f"data: {json.dumps({'message': 'No articles found for the selected criteria. Try expanding the date range or changing categories.'})}\n\n"
            return
        
        yield f"data: {json.dumps({'message': f'Found {len(db_articles)} articles. Ranking by relevance...', 'progress': 40})}\n\n"

        # --- 4. Rank Articles ---
        ranked_articles, scores = rank_articles(db_articles, keywords, negative_keywords)
        yield f"data: {json.dumps({'message': 'Ranking complete. Generating summaries...', 'progress': 60})}\n\n"

        # --- 5. Generate Summaries and Stream Results ---
        total_articles = len(ranked_articles)
        for i, (article, score) in enumerate(zip(ranked_articles, scores)):
            current_progress = 60 + int((i + 1) / total_articles * 40)
            yield f"data: {json.dumps({'message': f'Generating summary {i+1} of {total_articles}...', 'progress': current_progress})}\n\n"

            summary, insight = generate_insight_and_summary(article['abstract'])
            
            # Create article data payload for the client
            result_data = {
                'doi': article['doi'],
                'title': article['title'],
                'authors': article['authors'],
                'url': f"https://www.biorxiv.org/content/{article['doi']}",
                'similarity_score': f"{score:.4f}",
                'summary': summary,
                'insight': insight
            }
            yield f"data: {json.dumps(result_data)}\n\n"
            
        yield f"data: {json.dumps({'message': 'Process complete!', 'progress': 100})}\n\n"

    return Response(stream_with_context(stream_events()), content_type='text/event-stream')

@app.route('/clear', methods=['POST'])
def clear_session():
    """Clears all user settings from the session."""
    session.clear()
    return redirect(url_for('home'))

@app.route('/vote', methods=['POST'])
def vote():
    """Registers an upvote or downvote for an article."""
    data = request.get_json()
    doi = data.get('doi')
    vote_type = data.get('type')

    if not doi or vote_type not in ['upvote', 'downvote']:
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Check if the vote is the same as the one being cast
        current_vote = cursor.execute('SELECT vote_type FROM votes WHERE doi = ?', (doi,)).fetchone()
        
        if current_vote and current_vote['vote_type'] == vote_type:
            # If user clicks the same vote button again, remove the vote
            cursor.execute('DELETE FROM votes WHERE doi = ?', (doi,))
            new_vote_state = None
        else:
            # Insert or update the vote
            cursor.execute('''
                INSERT INTO votes (doi, vote_type) VALUES (?, ?)
                ON CONFLICT(doi) DO UPDATE SET vote_type = excluded.vote_type;
            ''', (doi, vote_type))
            new_vote_state = vote_type
            
        conn.commit()

    return jsonify({'status': 'success', 'doi': doi, 'newState': new_vote_state})

@app.route('/vote_status', methods=['GET'])
def get_vote_status():
    """Returns the vote status for all articles."""
    with get_db_connection() as conn:
        votes = conn.execute('SELECT doi, vote_type FROM votes').fetchall()
    return jsonify({row['doi']: row['vote_type'] for row in votes})

if __name__ == "__main__":
    init_db()  # Ensure DB is created on first run
    app.run(debug=True, port=5000)