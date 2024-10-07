from flask import Flask, render_template, request, redirect, url_for, send_file
import praw
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from wordcloud import WordCloud
import joblib
import os
import base64

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('models/sentiment_model.pkl')

# Reddit API credentials
reddit = praw.Reddit(
    client_id='QtW5i86wx_goa7ojeKNWQQ',
    client_secret='-fl2QLqMuinZfdWDTt2JrUoFD8E7zg',
    user_agent='sentimental_analysis_project:v1.0.0 (by u/Realistic-Youth-4325)'
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/button', methods=['GET', 'POST'])
def button():
    if request.method == 'POST':
        url = request.form['url']
        
        # Extract post_id from URL
        post_id = None
        if 'comments/' in url:
            parts = url.split('comments/')
            if len(parts) > 1:
                post_id = parts[1].split('/')[0]
        elif '/r/' in url:
            parts = url.split('/comments/')
            if len(parts) > 1:
                post_id = parts[1].split('/')[0]
        elif '/comments/' in url:
            parts = url.split('/comments/')
            if len(parts) > 1:
                post_id = parts[1].split('/')[0]
        
        if post_id:
            try:
                post = reddit.submission(id=post_id)
                
                # Analyze comments
                sid = SentimentIntensityAnalyzer()
                data = []
                post.comments.replace_more(limit=0)
                for comment in post.comments.list():
                    sentiment = sid.polarity_scores(comment.body)
                    comment_time = pd.to_datetime(comment.created_utc, unit='s')
                    data.append([comment.body, sentiment['compound'], comment_time])
                
                df = pd.DataFrame(data, columns=['Comment', 'Sentiment', 'Time'])
                df['Sentiment_Category'] = df['Sentiment'].apply(lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral'))
                df['Cleaned_Comment'] = df['Comment'].str.lower()
                
                # Save CSV file
                csv_path = 'static/reddit_post_sentiment_analysis.csv'
                df.to_csv(csv_path, index=False)
                
                return redirect(url_for('analysis'))
            except Exception as e:
                return f"Error occurred: {e}"
        else:
            return "Invalid URL or Post ID"
        
    return render_template('button.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/download')
def download():
    csv_path = 'static/reddit_post_sentiment_analysis.csv'
    return send_file(csv_path, as_attachment=True)

@app.route('/plot')
def plot():
    csv_path = 'static/reddit_post_sentiment_analysis.csv'
    df = pd.read_csv(csv_path)
    
    def plot_visualizations():
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        df['Sentiment_Category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
        plt.title('Sentiment Distribution')
        plt.ylabel('')
        
        plt.subplot(2, 2, 2)
        sns.countplot(x='Sentiment_Category', data=df, palette='viridis')
        plt.title('Sentiment Count')
        
        df['Word_Count'] = df['Comment'].apply(lambda x: len(x.split()))
        plt.subplot(2, 2, 3)
        sns.histplot(df['Word_Count'], bins=30, kde=True)
        plt.title('Word Count Distribution')
        
        plt.subplot(2, 2, 4)
        sns.scatterplot(x='Word_Count', y='Sentiment', hue='Sentiment_Category', data=df, palette='viridis')
        plt.title('Sentiment Scores vs Word Count')
        
        plt.tight_layout()
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        return img

    img = plot_visualizations()
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    
    return render_template('plot.html', img_data=img_base64)

@app.route('/wordcloud')
def wordcloud():
    df = pd.read_csv('static/reddit_post_sentiment_analysis.csv')
    positive_text = ' '.join(df[df['Sentiment_Category'] == 'Positive']['Cleaned_Comment'])
    negative_text = ' '.join(df[df['Sentiment_Category'] == 'Negative']['Cleaned_Comment'])
    
    def generate_wordcloud(text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        img = BytesIO()
        wordcloud.to_image().save(img, format='png')
        img.seek(0)
        return img

    pos_wordcloud_img = generate_wordcloud(positive_text)
    neg_wordcloud_img = generate_wordcloud(negative_text)
    
    pos_wordcloud_base64 = base64.b64encode(pos_wordcloud_img.getvalue()).decode('utf-8')
    neg_wordcloud_base64 = base64.b64encode(neg_wordcloud_img.getvalue()).decode('utf-8')
    
    return render_template('wordcloud.html', pos_wordcloud_data=pos_wordcloud_base64, neg_wordcloud_data=neg_wordcloud_base64)

    
    return render_template('wordcloud.html', pos_wordcloud_data=pos_wordcloud_base64, neg_wordcloud_data=neg_wordcloud_base64)


if __name__ == '__main__':
    app.run(debug=True)
