from transformers import pipeline
import re

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
