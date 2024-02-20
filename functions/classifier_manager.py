from transformers import pipeline, AutoTokenizer

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
