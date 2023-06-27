from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import numpy as np
import re
import joblib
import nltk
from google_play_scraper import Sort, reviews_all, app as application

app = Flask(__name__)

model = joblib.load('decision_tree_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def predict_text(text):
    def process_text(review_text):
        stop = set(nltk.corpus.stopwords.words('english'))
        if review_text is None or review_text == '':
            review_text = ''
        review_text = review_text.lower()
        review_text = re.sub(r'\W',' ', review_text)
        
        review_text = re.sub(r'\bnot\b', 'not_', review_text)
        review_text = re.sub(r'\bno\b', 'no_', review_text)
        
        words = review_text.split()
        review_text = ' '.join([word for word in words if word not in stop])

        vectorized_text = vectorizer.transform([review_text])
        
        return vectorized_text
    
    processed_text = process_text(text)
    predictions = model.predict(processed_text)
    return predictions[0]

def fetch_app(url):
    url = url.strip()
    app_id = re.search(r'id=([\w.]+)', url).group(1)

    app_details = application(app_id)
    
    #SET MAX (memory issues)
    if app_details['reviews'] > 5000:
        app_details['reviews'] = 5000
    
    return app_details
    
def classify_app(url):
    app_details = fetch_app(url)
    results = reviews_all(
        app_details["appId"],
        lang='en',
        sort=Sort.NEWEST
    )
    
    label = ['Positive', 'Neutral', 'Negative']
    sentimentCount = [0, 0, 0]
    scraped_reviews = ([],[])
    total_scraped_reviews = 0
    for r in results:
        if r['content'] is not None:
            predicted_text = predict_text(r['content'])
            scraped_reviews[0].append(r['content'])
            scraped_reviews[1].append(predicted_text)
            total_scraped_reviews += 1
            if(predicted_text.lower() == 'positive'):
                sentimentCount[0] += 1
            else:
                if(predicted_text.lower() == 'negative'):
                    sentimentCount[2] += 1
                else:
                    sentimentCount[1] += 1
    
    return scraped_reviews, app_details, total_scraped_reviews, label, sentimentCount
    
@app.route("/")
def main():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/single_text', methods=['GET','POST'])
def get_single_output():
    if request.method == 'POST':
        review_text = request.form["review_text"]
        prediction = predict_text(review_text)
        return jsonify({'prediction': prediction, 'review_text': review_text})

@app.route('/review', methods=['GET','POST'])
def get_url_output():
    if request.method == 'POST':
        url = request.form["url"]
        scraped_reviews, app_details, total_scraped_reviews, label, sentimentCount = classify_app(url)
        reviews = scraped_reviews[0]
        sentiments = scraped_reviews[1]
        return jsonify({'reviews': reviews, 'sentiments': sentiments, 
                        'app_details': app_details, 'total_scraped_reviews': total_scraped_reviews, 
                        'label': label, 'sentimentCount': sentimentCount})

if __name__ == '__main__':
    app.run(port=3000, debug=True)
