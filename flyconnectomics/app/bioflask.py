from flask import Flask, render_template, request, Response, stream_with_context, session, redirect, url_for
import requests
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import sqlite3
import numpy as np
import os
import json
import torch

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'template'))
app.secret_key = 'your_secret_key'

# Database file
DATABASE = 'articles.db'

# Initialize the database
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                doi TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                abstract TEXT,
                category TEXT,
                pub_date TEXT
            );
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS votes (
                doi TEXT PRIMARY KEY,
                vote_type TEXT CHECK(vote_type IN ('upvote', 'downvote', NULL)),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        conn.commit()

def get_cached_date_range():
    """
    Get the earliest and latest publication dates in the database.
    """
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT MIN(pub_date), MAX(pub_date) FROM articles')
        result = cursor.fetchone()
        return result[0], result[1]  # (earliest_date, latest_date)

def fetch_missing_data(start_date, end_date):
    """
    Fetch missing data from BioRxiv for the given date range.
    """
    base_url = 'https://api.biorxiv.org/details/biorxiv'
    pagination_cursor = 0
    batch_size = 100
    all_articles = []

    while True:
        try:
            response = requests.get(f'{base_url}/{start_date}/{end_date}/{pagination_cursor}')
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break

        data = response.json()
        if data['messages'][0]['status'] != 'ok':
            break

        batch_articles = data.get('collection', [])
        if not batch_articles:
            break

        # Add publication date to each article
        for article in batch_articles:
            article['pub_date'] = article['date']

        all_articles.extend(batch_articles)
        pagination_cursor += batch_size

    # Save new articles to the database
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        for article in all_articles:
            try:
                cursor.execute('''
                    INSERT INTO articles (doi, title, authors, abstract, category, pub_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (article['doi'], article['title'], article['authors'], article['abstract'], article['category'], article['pub_date']))
            except sqlite3.IntegrityError:
                pass
        conn.commit()

    return all_articles

# Function to extract insights from abstracts
def extract_insights_from_abstract(abstract):
    questions = [
        "What are the main objective of this study?",
        "What is the main result of this study?",
        "What is the significance of this study?",
    ]
    
    insights = []
    for question in questions:
        try:
            result = qa_model(question=question, context=abstract)
            insights.append(result['answer'])
        except Exception as e:
            insights.append(f"Error retrieving answer: {e}")
    
    return "Goal: " + insights[0] + " | Results: " + insights[1] + " | Significance: " + insights[2]

init_db()

# Determine if CUDA is available for GPU usage
device = 0 if torch.cuda.is_available() else -1

# Initialize the summarization and embedding models
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
summarizer = pipeline("summarization", model="Falconsai/text_summarization", device=device, clean_up_tokenization_spaces=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cleanup old data (delete articles older than 6 months)
def cleanup_old_data():
    cutoff_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM articles WHERE pub_date < ?', (cutoff_date,))
        conn.commit()
        print(f"Cleanup completed: Articles older than {cutoff_date} were removed.")

# Scheduler setup
scheduler = BackgroundScheduler()
scheduler.add_job(func=cleanup_old_data, trigger="interval", days=1)  # Run daily
scheduler.start()

# Ensure the scheduler shuts down cleanly on exit
atexit.register(lambda: scheduler.shutdown())

@app.route('/', methods=['GET', 'POST'])
def home():
    # Retrieve tags from session and ensure they are lists
    keywords = session.get('keywords', [])  # Default to empty list if none
    negative_keywords = session.get('negative_keywords', [])  # Default to empty list if none
    categories = session.get('categories', []) if session.get('categories') else []
    period_days = session.get('period_days', '')

    return render_template(
        'index.html',
        keywords=keywords,
        negative_keywords=negative_keywords,
        categories=categories,
        period_days=period_days
    )

@app.route('/generate_summaries', methods=['GET'])
def generate_summaries():
    # Get keywords and negative keywords as lists from request parameters
    keywords = request.args.getlist('keywords')
    negative_keywords = request.args.getlist('negative_keywords')
    categories = request.args.getlist('categories') if request.args.get('categories') else []
    period_days = int(request.args.get('period_days', 0)) if request.args.get('period_days') else 0

    # Cache the tags in the session directly as lists
    session['keywords'] = keywords
    session['negative_keywords'] = negative_keywords
    session['categories'] = categories
    session['period_days'] = period_days

    print(f"Keywords: {keywords}")
    print(f"Negative Keywords: {negative_keywords}")
    print(f"Categories: {categories}")
    print(f"Period (days): {period_days}")


    # Calculate start and end dates
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_days)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Check the cache
    earliest_cached, latest_cached = get_cached_date_range()
    print(f"Earliest cached date: {earliest_cached}, Latest cached date: {latest_cached}")

    if earliest_cached is None or latest_cached is None:
        # Cache is empty, fetch the entire range
        print("Cache is empty, fetching all data.")
        fetch_missing_data(start_str, end_str)
    else:
        # Fetch missing data
        if start_str < earliest_cached:
            print("Fetching missing data before the cached range.")
            fetch_missing_data(start_str, earliest_cached)
        if end_str > latest_cached:
            print("Fetching missing data after the cached range.")
            fetch_missing_data(latest_cached, end_str)

    @stream_with_context
    def stream_summaries():
        # Step 1: Fetch articles from the database
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT doi, title, authors, abstract, category, pub_date
                FROM articles
                WHERE pub_date BETWEEN ? AND ?
            ''', (start_str, end_str))
            articles = [
                {'doi': row[0], 'title': row[1], 'authors': row[2], 'abstract': row[3], 'category': row[4], 'pub_date': row[5]}
                for row in cursor.fetchall()
            ]

        # If no articles found in database, fetch from API
        if not articles:
            base_url = f'https://api.biorxiv.org/details/biorxiv/{start_str}/{end_str}'
            pagination_cursor = 0
            batch_size = 100

            while True:
                try:
                    response = requests.get(f'{base_url}/{pagination_cursor}')
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    yield f"data: {json.dumps({'status': 'Error fetching articles', 'error': str(e)})}\n\n"
                    return

                data = response.json()
                if data['messages'][0]['status'] != 'ok':
                    break

                batch_articles = data.get('collection', [])
                if not batch_articles:
                    break

                # Add publication date to each article
                for article in batch_articles:
                    article['pub_date'] = article['date']

                # Save articles to the database
                with sqlite3.connect(DATABASE) as conn:
                    db_cursor = conn.cursor()
                    for article in batch_articles:
                        try:
                            db_cursor.execute('''
                                INSERT INTO articles (doi, title, authors, abstract, category, pub_date)
                                VALUES (?, ?, ?, ?, ?, ?)
                            ''', (article['doi'], article['title'], article['authors'], article['abstract'], article['category'], article['pub_date']))
                        except sqlite3.IntegrityError:
                            pass
                    conn.commit()

                articles.extend(batch_articles)
                pagination_cursor += batch_size

                # Send progress update for data fetching
                yield f"data: {json.dumps({'progress': 'Fetching data from API', 'batch_fetched': len(batch_articles), 'total_fetched': len(articles)})}\n\n"

        # Step 2: Filter articles by the user-specified categories
        filtered_articles = [
            article for article in articles
            if any(cat.lower() in article.get('category', '').lower() for cat in categories)
        ]
        yield f"data: {json.dumps({'progress': 'Filtering articles by category', 'filtered_count': len(filtered_articles)})}\n\n"

        # Step 3: Calculate summaries and similarities
        abstracts = [article['abstract'] for article in filtered_articles]
        valid_articles = filtered_articles
        abstract_embeddings = model.encode(abstracts, convert_to_tensor=True)
        weighted_embeddings = None
        for i, keyword in enumerate(keywords):
            embedding = model.encode(keyword, convert_to_tensor=True) * (len(keywords) - i)
            weighted_embeddings = embedding if weighted_embeddings is None else weighted_embeddings + embedding
        for negative_keyword in negative_keywords:
            negative_embedding = model.encode(negative_keyword, convert_to_tensor=True)
            weighted_embeddings -= negative_embedding
        weighted_embeddings = weighted_embeddings / torch.norm(weighted_embeddings)
        cosine_scores = util.cos_sim(weighted_embeddings, abstract_embeddings)[0].cpu()

        # Sort articles by similarity
        sorted_indices = np.argsort(-cosine_scores)
        sorted_articles = [valid_articles[idx] for idx in sorted_indices]
        sorted_scores = [cosine_scores[idx].item() for idx in sorted_indices]

        # Send progress for each summary generated
        total_articles = len(sorted_articles)
        for i, (article, score) in enumerate(zip(sorted_articles, sorted_scores)):
            insight = extract_insights_from_abstract(article['abstract'])
            summary = summarizer(article['abstract'], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            progress = int((i + 1) / total_articles * 100)
            data = {
                'title': article['title'],
                'authors': article['authors'],
                'url': f"https://www.biorxiv.org/content/{article['doi']}",
                'similarity_score': f"{score:.4f}",
                'summary': summary + ' | ' + insight,
                'progress': progress,
                'message': f'Processing article {i+1} of {total_articles}'
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        # Indicate that the stream is complete
        yield f"data: {json.dumps({'progress': 100, 'message': 'All summaries processed successfully.'})}\n\n"

    return Response(stream_summaries(), content_type='text/event-stream')

@app.route('/clear', methods=['POST'])
def clear():
    session.pop('keywords', None)
    session.pop('negative_keywords', None)
    session.pop('categories', None)
    session.pop('period_days', None)
    return redirect(url_for('home'))

@app.route('/vote', methods=['POST'])
def vote():
    doi = request.args.get('doi')
    vote_type = request.args.get('type')  # Either 'upvote', 'downvote', or null

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        # Check the current vote status
        cursor.execute('SELECT vote_type FROM votes WHERE doi = ?', (doi,))
        current_vote = cursor.fetchone()

        if current_vote:
            current_vote_type = current_vote[0]
            # If the user clicks the same vote, toggle it off (set to null)
            if current_vote_type == vote_type:
                cursor.execute('UPDATE votes SET vote_type = NULL WHERE doi = ?', (doi,))
            else:
                # Update the vote type
                cursor.execute('UPDATE votes SET vote_type = ? WHERE doi = ?', (vote_type, doi))
        else:
            # Insert a new vote
            cursor.execute('INSERT INTO votes (doi, vote_type) VALUES (?, ?)', (doi, vote_type))

        conn.commit()

    return "Vote registered", 200

@app.route('/vote_status', methods=['GET'])
def vote_status():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT doi, vote_type FROM votes')
        data = cursor.fetchall()

    vote_status = {row[0]: row[1] for row in data}
    return json.dumps(vote_status)

if __name__ == "__main__":
    app.run(debug=True, port=12345)
