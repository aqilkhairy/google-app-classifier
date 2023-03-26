import joblib
import tkinter as tk
from google_play_scraper import app, Sort, reviews

from sklearn.feature_extraction.text import TfidfVectorizer

window = tk.Tk()
window.title('Google Play Store App Classifier')
input_box = tk.Entry(window)
input_box.pack()

#global variables
app_id = ''
total_reviews = 0
name = ''
logo = ''

def fetch_app():
    global app_id, name, logo, total_reviews
    url = input_box.get()
    
    app_id = url.split('=')[1]

    # Scrape app information
    app_details = app(app_id)
    name = app_details['title']
    logo = app_details['icon']
    total_reviews = app_details['reviews']
    if total_reviews > 5000:
        total_reviews = 5000
    print(total_reviews)

def classify_app():
    global app_id, name, logo, total_reviews
    # Scrape reviews
    results, continuation_token = reviews(
        app_id,
        lang='en',
        count=total_reviews,
        sort=Sort.NEWEST
    )
    
    scraped_reviews = []
    empty_reviews = 0
    for r in results:
        if r['content'] is not None:
            scraped_reviews.append(r['content'])
        else:
            empty_reviews += 1
    
    # Classify app using trained model
    clf = joblib.load('decision_tree_model.joblib')
    # Transform the textual data into numerical features using TF-IDF
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    scraped_reviews = tfidf_vectorizer.transform(scraped_reviews)
    result = clf.predict(scraped_reviews)
    
    positive_count = 0
    for score in result:
        if score >= 3:
            positive_count += 1
    positive_percentage = (positive_count/total_reviews) * 100
    
    # Display result in GUI
    result_label.config(text=f'{name}\n{logo}\nPositive Percentage: {positive_percentage}')
    
fetch_button = tk.Button(window, text='Fetch Application', command=fetch_app)
fetch_button.pack()
classify_button = tk.Button(window, text='Classify', command=classify_app)
classify_button.pack()
result_label = tk.Label(window)
result_label.pack()
window.mainloop()
