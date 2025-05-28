from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import string
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import re

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load data and model
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Load embeddings and metadata
try:
    embeddings = np.load('models/arxiv_embeddings.npy')
    df = pd.read_csv('models/arxiv_metadata.csv')
    # Ensure all text fields are strings
    df['title'] = df['title'].astype(str)
    df['abstract'] = df['abstract'].astype(str)
    df['authors'] = df['authors'].astype(str)
    df['categories'] = df['categories'].astype(str)
except Exception as e:
    print(f"Error loading data: {e}")
    raise

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def search_by_id(paper_id):
    """Search by exact paper ID match"""
    results = df[df['id'].str.contains(paper_id, case=False, regex=False)]
    return results.to_dict('records')

def search_by_author(author_name):
    """Search by author name"""
    results = df[df['authors'].str.contains(author_name, case=False, regex=False)]
    return results.to_dict('records')

def search_by_semantic(query, top_k=5):
    """Semantic search using embeddings"""
    clean_query = clean_text(query)
    query_embedding = model.encode([clean_query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        paper = df.iloc[idx]
        results.append({
            'id': paper['id'],
            'title': paper['title'],
            'authors': paper['authors'],
            'abstract': paper['abstract'],
            'categories': paper['categories'],
            'score': float(scores[idx])
        })
    return results

@app.route('/')
def home():
    return render_template('api_docs.html')

@app.route('/search', methods=['GET', 'POST'])
def search_papers():
    """Unified search endpoint"""
    if request.method == 'GET':
        # Handle GET requests with query parameters
        query = request.args.get('query', '')
        search_type = request.args.get('type', 'semantic')  # semantic, id, author
        top_k = int(request.args.get('top_k', 5))
    else:
        # Handle POST requests with JSON body
        data = request.get_json()
        query = data.get('query', '')
        search_type = data.get('type', 'semantic')
        top_k = int(data.get('top_k', 5))
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    try:
        # Determine search type
        if search_type == 'id':
            results = search_by_id(query)
        elif search_type == 'author':
            results = search_by_author(query)
        else:  # semantic search
            results = search_by_semantic(query, top_k)
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/paper/<paper_id>')
def get_paper(paper_id):
    """Get single paper by ID"""
    paper = df[df['id'] == paper_id]
    if paper.empty:
        return jsonify({'error': 'Paper not found'}), 404
    return jsonify(paper.iloc[0].to_dict())

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'entries': len(df)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    