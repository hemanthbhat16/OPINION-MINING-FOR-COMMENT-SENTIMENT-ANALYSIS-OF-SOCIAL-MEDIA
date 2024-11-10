from collections import Counter
import praw
import string
import random
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
from praw.exceptions import InvalidURL

# Initialize the stemmer
port_stem = PorterStemmer()

# Preprocessing function
def preprossing(content):
    if isinstance(content, list):
        return [preprossing(comment) for comment in content]
    
    if isinstance(content, float):
        content = ''
    
    # Remove punctuation and non-alphabetic characters
    content = re.sub(f"[{re.escape(string.punctuation)}]", '', content)  # Remove punctuation
    content = re.sub('[^a-zA-Z]', ' ', content)  # Remove non-alphabetic characters
    content = content.lower().split()
    
    # Stem and remove stopwords
    content = [port_stem.stem(w) for w in content if w not in stopwords.words('english')]
    return ' '.join(content)

# Prediction function
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

# Main function to process the Reddit post and predict sentiment
def main(URL):
    # Create a Reddit instance with your app credentials
    reddit = praw.Reddit(
        client_id="jjfi3QgyIrd7WpGkP3zQPA",
        client_secret="-1-OdFADrw5aKKdbLLAyVuFcnpdN7Q",
        user_agent="shaasar",
    )

    try:
        # Get the post you want to scrape comments from
        post = reddit.submission(url=URL)
    except InvalidURL:
        print("Invalid URL. Please provide a valid Reddit submission URL.")
        return [], {}  # Return empty lists if URL is invalid

    # Get the top-level comments
    top_level_comments = list(post.comments)

    # Prepare a list to store the comments data
    comments_data = []

    for comment in top_level_comments:
        if isinstance(comment, praw.models.Comment):
            # Append the author and body of each comment to the list
            comments_data.append([comment.author, comment.body])

    print(f"Total comments fetched: {len(comments_data)}")
    
    # Extract only the body of comments
    all_comments = [comment[1] for comment in comments_data]
    
    # Preprocess the comments
    preprocessed_comments = preprossing(all_comments)
    
    # Randomly sample 1000 comments if more than 1000 are available
    if len(preprocessed_comments) > 1000:
        random_comments = random.sample(preprocessed_comments, 1000)
    else:
        random_comments = preprocessed_comments
    
    # Load the pre-trained model and vectorizer
    with open('stacking_ensemble_model.sav', 'rb') as file:
        loaded_model = pickle.load(file)
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    # Predict sentiment
    predictions = predict(loaded_model, random_comments, vectorizer)
    
    # Get word frequency
    word_frequency = get_word_frequency(preprocessed_comments)
    
    return predictions, word_frequency

if __name__ == "__main__":
    url = input("Enter Reddit URL: ")
    predictions, word_frequency = main(url)
    if predictions and word_frequency:
        print("Predictions:", predictions)
        print("Word Frequency:", word_frequency)
    else:
        print("No valid predictions or word frequencies returned.")
