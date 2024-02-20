import streamlit as st
from transformers import pipeline, AutoTokenizer
import json
import re
import pandas as pd 
import base64

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
            zero_shot_prob = zero_shot_classification['scores'][0] * 100  # Convert to percentage
            return {
                "tweet": tweet,
                "zero_shot_classification": zero_shot_sentiment,
                "zero_shot_prob": f"{zero_shot_prob:.2f}%"  # Format as percentage with 2 decimal places
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


def main():
    st.title("Tweet Classification App")

    # Option to upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        # Read the CSV file
        tweets_df = pd.read_csv(uploaded_file)
        with open('section_data/label_data_og.json', 'r') as json_file:
            data = json.load(json_file)
        # Analyze each tweet in the CSV file
        analysis_results = []
        for index, row in tweets_df.iterrows():
            tweet_text = row['tweet']
            analyzer = TweetSentimentAnalyzer()
            analysis_result = analyzer.analyze_tweet(tweet_text)
            analysis_results.append(analysis_result)
            manager = ZeroShotClassifierManager(data)
            highest_scores = manager.classify_tweets(tweet_text)
            best_key, best_label, overall_best_label, overall_best_score = manager.determine_best_key(tweet_text, highest_scores)
            
            # Update the analysis result with the best key, best label, overall best label, and overall best score
            analysis_result['Best Key'] = best_key
            analysis_result['Best Label within the Key'] = best_label
            analysis_result['Overall Best Label'] = overall_best_label
            analysis_result['Overall Best Score'] = f"{overall_best_score * 100:.2f}%"  # Format as percentage with 2 decimal places
        
        # Display the analysis results in a table
        st.write("Analysis Results:")
        st.table(pd.DataFrame(analysis_results))

        # Option to download the updated CSV file
        if st.button("Download Updated CSV"):
            updated_df = tweets_df.copy()
            updated_df['Zero-Shot Classification'] = [result['zero_shot_classification'] for result in analysis_results]
            updated_df['Zero-Shot Classification Probability'] = [result['zero_shot_prob'] for result in analysis_results]
            updated_df['Best Key'] = [result['Best Key'] for result in analysis_results]
            updated_df['Best Label within the Key'] = [result['Best Label within the Key'] for result in analysis_results]
            updated_df['Overall Best Label'] = [result['Overall Best Label'] for result in analysis_results]
            updated_df['Overall Best Score'] = [result['Overall Best Score'] for result in analysis_results]
            
            # Save the updated DataFrame to a new CSV file
            csv_data = updated_df.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="updated_tweets.csv">Download Updated CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
