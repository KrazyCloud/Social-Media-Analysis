import streamlit as st
from transformers import pipeline, AutoTokenizer
import json
import re
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from torch.nn.functional import softmax
import pandas as pd 
import matplotlib.pyplot as plt

class TweetSentimentAnalyzer:
    def __init__(self):
        self.zero_shot_pipeline = pipeline("zero-shot-classification")

    def preprocess_tweet(self, tweet):
        tweet = re.sub(r'[^\w\s#@]', '', tweet)
        tweet = re.sub(r'http\S+', '', tweet)
        return tweet.strip()


    def analyze_tweet(self, tweet):
        try:
            processed_tweet = self.preprocess_tweet(tweet)

            zero_shot_classification = self.zero_shot_pipeline(processed_tweet, candidate_labels=["positive", "negative", "neutral"])
            zero_shot_sentiment = zero_shot_classification['labels'][0]
            zero_shot_prob = zero_shot_classification['scores'][0]

            return {
                "tweet": tweet,
                "zero_shot_classification": zero_shot_sentiment,
                "zero_shot_prob": zero_shot_prob
            }
        except Exception as e:
            return {"error": str(e)}

class ZeroShotClassifier:
    def __init__(self, model_name):
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def classify(self, tweet, labels):
        results = self.classifier(tweet, labels, tokenizer=self.tokenizer)
        top_label = results["labels"][0]
        top_score = results["scores"][0]
        return top_label, top_score

class ZeroShotClassifierManager:
    def __init__(self, data):
        self.data = data
        self.classifier = ZeroShotClassifier("facebook/bart-large-mnli")
    
    def classify_tweets(self, tweet):
        highest_scores = {}
        for key, labels in self.data.items():
            highest_score = 0
            highest_label = None
            for label in labels:
                _, score = self.classifier.classify(tweet, [label])
                if score > highest_score:
                    highest_score = score
                    highest_label = label
            highest_scores[key] = {"label": highest_label, "score": highest_score}
        return highest_scores
    
    def determine_best_key(self, tweet, highest_scores):
        best_key = max(highest_scores, key=lambda k: highest_scores[k]["score"])
        best_label = highest_scores[best_key]["label"]
        overall_best_label, overall_best_score = self.classifier.classify(tweet, [best_label])
        return best_key, best_label, overall_best_label, overall_best_score

def generate_pie_chart(highest_scores):
    # Extract keys, labels, and scores
    keys = list(highest_scores.keys())
    labels = [f"{key}: {entry['label']}" for key, entry in highest_scores.items()]
    scores = [entry["score"] for entry in highest_scores.values()]

    # Plotting the pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(scores, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.set_title('Distribution of Highest Scores by Category')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

def plot_bar_chart(highest_scores):
    # Extracting data for the bar chart
    keys = list(highest_scores.keys())
    labels = [score["label"] for score in highest_scores.values()]
    scores = [score["score"] for score in highest_scores.values()]

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(keys, scores, color='skyblue')
    ax.set_xlabel('Keys')
    ax.set_ylabel('Scores')
    ax.set_title('Highest Label and Score for Each Key')
    ax.set_xticks(keys)
    ax.set_xticklabels(keys, rotation=45)

    # Annotate each bar with its percentage value
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(f'{score*100:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    st.pyplot(fig)


def main():
    st.title("Tweet Classification App")
    tweet = st.text_area("Enter your tweet:", height=200)
    if st.button("Analyse"):
        # Load label_data from JSON
        with open('section_data/label_data_og.json', 'r') as json_file:
            data = json.load(json_file)

        analyzer = TweetSentimentAnalyzer()
        analysis_result = analyzer.analyze_tweet(tweet)
        manager = ZeroShotClassifierManager(data)
        highest_scores = manager.classify_tweets(tweet)
        best_key, best_label, overall_best_label, overall_best_score = manager.determine_best_key(tweet, highest_scores)
        zero_shot_prob_percentage = f"{analysis_result['zero_shot_prob'] * 100:.2f}%"

        table_data = {
            "Aspect": ["Tweet", "Zero-Shot Classification", 
                       "Zero-Shot Classification Probability",
                       "Best key", "Best label within the key", "Overall best label", "Overall best score"],
            "Value": [analysis_result['tweet'],
                      analysis_result['zero_shot_classification'],
                      zero_shot_prob_percentage,
                      best_key, best_label, overall_best_label, f"{overall_best_score * 100:.2f}%"]
        }

        # Display the results in a single table without indexing
        st.table(pd.DataFrame(table_data).set_index("Aspect", drop=True))

        # Generate and display visualization
        generate_pie_chart(highest_scores)
        plot_bar_chart(highest_scores)

if __name__ == "__main__":
    main()
