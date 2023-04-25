import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('stopwords')

# Load the trained model and preprocessing objects
classifier = load_model('trained_model.h5')
cv = pickle.load(open('count-Vectorizer.pkl','rb'))
sc = pickle.load(open('Standard-Scaler.pkl','rb'))

# Function to perform sentiment analysis
def predict_sentiment(df, column_name):
    input_reviews = df[column_name].tolist()
    input_reviews = [re.sub(pattern='[^a-zA-Z]', repl=' ', string=review) for review in input_reviews]
    input_reviews = [review.lower() for review in input_reviews]
    input_reviews = [review.split() for review in input_reviews]
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    input_reviews = [[word for word in review_words if not word in stop_words] for review_words in input_reviews]
    ps = PorterStemmer()
    input_reviews = [[ps.stem(word) for word in review_words] for review_words in input_reviews]
    input_reviews = [' '.join(review_words) for review_words in input_reviews]
    input_X = cv.transform(input_reviews).toarray()
    input_X = sc.transform(input_X)
    pred = classifier.predict(input_X)
    pred = (pred > 0.5)
    sentiment = ['Positive review' if p else 'Negative review' for p in pred]
    return sentiment



# Function to show the analytics in a separate tab
def show_analytics(df, column_name):
    # Apply sentiment analysis to specified column
    sentiments = predict_sentiment(df, column_name)
    
    # Get the count of reviews and positive/negative reviews
    total_reviews = len(sentiments)
    positive_reviews = sentiments.count('Positive review')
    negative_reviews = sentiments.count('Negative review')
    
    # Print the count of reviews and positive/negative reviews
    st.write(f"Total number of reviews: {total_reviews}")
    st.write(f"Number of positive reviews: {positive_reviews}")
    st.write(f"Number of negative reviews: {negative_reviews}")
    
    # Plot the sentiment analysis results using matplotlib
    fig, ax = plt.subplots()
    ax.bar(['Positive', 'Negative'], [positive_reviews, negative_reviews], color=['blue', 'orange'])
    ax.set_title('Sentiment Analysis Results')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)




# Main function to run the app
def main():
    st.title('Student sentiment analysis')

    # Get the user inputs
    file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
    if file is not None:
        df = pd.read_excel(file)
        column_name = st.selectbox('Select column to analyze:', df.columns)
        st.write(df)
        
        # Show analytics in a separate tab on click of a button
        if st.button('Show Analytics'):
            show_analytics(df, column_name)


# Run the app
if __name__=='__main__':
    main()
