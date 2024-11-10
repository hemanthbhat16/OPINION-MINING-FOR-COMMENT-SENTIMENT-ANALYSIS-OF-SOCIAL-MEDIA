from collections import Counter
import string
import random
import re
import joblib
from apify_client import ApifyClient
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle

port_stem = PorterStemmer()



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

def main(url):
    # Initialize the ApifyClient with your API token
    client = ApifyClient("apify_api_Ycvtx0urqpIcDd0tdpWLH92iwObKj93IAATN")

    # Prepare the Actor input
    run_input = {
        "directUrls": [],
        "resultsLimit": 500000,
        "commentsMode": "EARLIEST"
    }

    # Add the entered URL to the directUrls list
    run_input["directUrls"].append(url)

    # Run the Actor and wait for it to finish
    run = client.actor("SbK00X0JYCPblD2wp").call(run_input=run_input)

    # Fetch and print Actor results from the run's dataset (if there are any)
    data_list = []

    # Iterate through the items in the dataset
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        
        text = item.get("text", "")

        likes_count = item.get("likesCount", 0)

        # Append the extracted data to the list
        data_list.append({
            "Comment": text,
            "likesCount": likes_count
        })
    
    print(len(data_list))
    all_comments = [comment['Comment'] for comment in data_list]
    
    preprocessed_comments=preprossing(all_comments)
    if len(preprocessed_comments) > 1000:
        random_comments = random.sample(preprocessed_comments, 1000)
    else:
        random_comments = preprocessed_comments
    
    with open('stacking_ensemble_model.sav', 'rb') as file:
        loaded_model = pickle.load(file)
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    predictions=predict(loaded_model,random_comments,vectorizer)

     # Get word frequency
    word_frequency = get_word_frequency(preprocessed_comments)
    
    return predictions, word_frequency


if __name__ == "__main__":
    instagram_url = input("Enter the Instagram post URL: ")
    main(instagram_url)