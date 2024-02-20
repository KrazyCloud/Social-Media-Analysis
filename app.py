import streamlit as st
import json
import pandas as pd 
import base64
from functions.tweet_senitment import TweetSentimentAnalyzer
from functions.classifier_manager import ZeroShotClassifierManager
from functions.visual_graphs import ChartGenerator

def main():
    st.title("Tweet Analysis")

    # Sidebar option for selecting the analysis mode
    analysis_mode = st.sidebar.radio("Select Analysis Mode", ("Tweet", "Tweet File"))

    if analysis_mode == "Tweet":

        # Input field for entering a single tweet
        tweet = st.text_area("Enter your tweet:", height=150)

        if st.button("Analyse"):
            # Load label_data from JSON
            with open('section_data/label_data_og.json', 'r') as json_file:
                data = json.load(json_file)

            analyzer = TweetSentimentAnalyzer()
            analysis_result = analyzer.analyze_tweet(tweet)
            manager = ZeroShotClassifierManager(data)
            highest_scores = manager.classify_tweets(tweet)
            best_key, best_label, overall_best_label, overall_best_score = manager.determine_best_key(tweet, highest_scores)

            # Display tweet and analysis results in a table
            table_data = {
                "Aspect": ["Tweet", "Sentiment", 
                            "Sentiment Accuracy",
                            "Section", "Section Label", "Overall Section label", "Section Accuracy"],
                "Result": [analysis_result['tweet'],
                        analysis_result['zero_shot_classification'],
                        analysis_result['zero_shot_prob'],
                        best_key, best_label, overall_best_label, f"{overall_best_score * 100:.2f}%"]
            }
            st.table(pd.DataFrame(table_data).set_index("Aspect", drop=True))

            # Generate and display visualization
            ChartGenerator.generate_pie_chart(highest_scores)
            ChartGenerator.plot_bar_chart(highest_scores)

    elif analysis_mode == "Tweet File":

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
                
                # Update the analysis result with the Section, best label, overall best label, and Section Accuracy
                analysis_result['Section'] = best_key
                analysis_result['Section Label'] = best_label
                analysis_result['Overall Best Label'] = overall_best_label
                analysis_result['Section Accuracy'] = f"{overall_best_score * 100:.2f}%"
            
            # Display the analysis results in a table
            st.write("Analysis Results:")
            st.table(pd.DataFrame(analysis_results))

            # Option to download the updated CSV file
            if st.button("Download Updated CSV"):
                updated_df = tweets_df.copy()
                updated_df['Sentiment'] = [result['zero_shot_classification'] for result in analysis_results]
                updated_df['Sentiment Accuracy'] = [result['zero_shot_prob'] for result in analysis_results]
                updated_df['Section'] = [result['Section'] for result in analysis_results]
                updated_df['Section Label'] = [result['Section Label'] for result in analysis_results]
                updated_df['Overall Best Label'] = [result['Overall Best Label'] for result in analysis_results]
                updated_df['Section Accuracy'] = [result['Section Accuracy'] for result in analysis_results]
                
                # Save the updated DataFrame to a new CSV file
                csv_data = updated_df.to_csv(index=False)
                b64 = base64.b64encode(csv_data.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="updated_tweets.csv">Download Updated CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
