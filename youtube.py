import string
import random
import re
import joblib
import pickle
from googleapiclient.discovery import build
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

port_stem = PorterStemmer()

def extract_video_id(url):
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

def get_comments_for_video(youtube, video_id):
    all_comments = []
    next_page_token = None

    while True:
        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            textFormat="plainText",
            maxResults=100
        )
        comment_response = comment_request.execute()

        for item in comment_response['items']:
            top_comment = item['snippet']['topLevelComment']['snippet']
            all_comments.append({
                'Timestamp': top_comment['publishedAt'],
                'Username': top_comment['authorDisplayName'],
                'VideoID': video_id,
                'Comment': top_comment['textDisplay'],
                'Date': top_comment['updatedAt'] if 'updatedAt' in top_comment else top_comment['publishedAt']
            })

        next_page_token = comment_response.get('nextPageToken')
        if not next_page_token:
            break

    return all_comments

def preprossing(content):
    if isinstance(content, list):
        return [preprossing(comment) for comment in content]
    
    if isinstance(content, float):
        content = ''
    
    content = re.sub(f"[{re.escape(string.punctuation)}]", '', content)  # Remove punctuation
    content = re.sub('[^a-zA-Z]', ' ', content)  # Remove non-alphabetic characters
    content = content.lower().split()
    
    content = [port_stem.stem(w) for w in content if w not in stopwords.words('english')]
    return ' '.join(content)

def predict(model, content_list, vectorizer):
    predictions = []
    for content in content_list:
        vectorized_comments = vectorizer.transform([content])
        prediction = model.predict(vectorized_comments)
        predictions.append(prediction[0])
    return predictions

def get_word_frequency(preprocessed_comments):
    all_words = ' '.join(preprocessed_comments).split()
    word_counts = Counter(all_words)
    return word_counts


def main(urll):
    api_key = 'AIzaSyAsghIwYp0lPrRe6nT-TN8G2bmQ3CzsncA'
    url = urll
    video_id = extract_video_id(url)
    print(video_id)
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    all_comments = []
    video_comments = get_comments_for_video(youtube, video_id)
    all_comments = [comment['Comment'] for comment in video_comments]
    
    preprocessed_comments = preprossing(all_comments)
    
    # Load model and vectorizer
    with open('stacking_ensemble_model.sav', 'rb') as file:
        loaded_model = pickle.load(file)
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    # Get predictions
    predictions = predict(loaded_model, preprocessed_comments, vectorizer)
    
    # Get word frequency
    word_frequency = get_word_frequency(preprocessed_comments)
    
    return predictions, word_frequency

if __name__ == "__main__":
    urrl = input("Enter URL: ")
    predictions, word_frequency = main(urrl)
    print("Predictions:", predictions)
    print("Word Frequency:", word_frequency)

