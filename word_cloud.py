import matplotlib.pyplot as plt
from wordcloud import WordCloud


def main(word_frequencies):
    # Create the word cloud
    wordcloud = WordCloud(width=800, height=600, background_color=None, mode='RGBA', colormap='Set2', random_state=42).generate_from_frequencies(word_frequencies)

    # Save the word cloud image
    wordcloud_path = 'static/wordcloud.png'  # Save in the static folder
    wordcloud.to_file(wordcloud_path)

if __name__ == "__main__":
    main()