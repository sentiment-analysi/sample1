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
def predict_sentiment(input_review):
    input_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=input_review)
    input_review = input_review.lower()
    input_review_words = input_review.split()
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    input_review_words = [word for word in input_review_words if not word in stop_words]
    ps = PorterStemmer()
    input_review = [ps.stem(word) for word in input_review_words]
    input_review = ' '.join(input_review)
    input_X = cv.transform([input_review]).toarray()
    input_X = sc.transform(input_X)
    pred = classifier.predict(input_X)
    pred = (pred > 0.5)
    if pred[0][0]:
        return "Positive review"
    else:
        return "Negative review"


# Function to show the analytics in a separate tab
def show_analytics(df):
    # List of columns to perform sentiment analysis on
    columns = ['question1', 'question2']

    # Concatenate the results of sentiment analysis into a single column
    df['Sentiment'] = ''
    for col in columns:
        df[col] = df[col].fillna('')
        df['Sentiment'] += df[col].apply(predict_sentiment) + ' '

    # Get the count of reviews and positive/negative reviews
    total_reviews = len(df)
    positive_reviews = len(df[df['Sentiment'].str.contains('Positive review')])
    negative_reviews = len(df[df['Sentiment'].str.contains('Negative review')])
    
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

    # Perform sentiment analysis and show the results
    if file is not None:
        df = pd.read_excel(file)
        st.write(df)
        
        # Show analytics in a separate tab on click of a button
        if st.button('Show Analytics'):
            show_analytics(df)

# Run the app
if __name__=='__main__':
    main()
