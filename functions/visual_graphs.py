import matplotlib.pyplot as plt
import streamlit as st

class ChartGenerator:

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
