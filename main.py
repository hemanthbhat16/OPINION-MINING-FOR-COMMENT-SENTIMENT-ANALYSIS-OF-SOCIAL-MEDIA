from flask import Flask, render_template, request, jsonify
import youtube as yt
import instagram as ins
import word_cloud as wc
import reddit as rd
from collections import Counter

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/youtube')
def youtube():
    return render_template('youtube.html')

@app.route('/reddit')
def reddit():
    return render_template('reddit.html')

@app.route('/instagram')
def instagram():
    return render_template('instagram.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    platform = data['platform']
    url = data['url']
    
    # Call the platform-specific main function
    if platform == 'instagram':
        predictions, word_frequency = ins.main(url)
    elif platform == 'youtube':
        predictions, word_frequency = yt.main(url)
    elif platform == 'reddit':
        predictions, word_frequency = rd.main(url)
    else:
        return jsonify({'error': 'Unknown platform'}), 400

    # Count the sentiment predictions
    counts = Counter(predictions)

    # Prepare the result including both sentiment counts and word frequency
    result = {
        'positive': counts.get(1, 0),
        'neutral': counts.get(0, 0),
        'negative': counts.get(-1, 0),
    }
    print(word_frequency)
    wc.main(word_frequency)
    # Return the result as a JSON response
    
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
