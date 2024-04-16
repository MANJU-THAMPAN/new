import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image,ImageOps
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('all')


# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("WomensClothingE-CommerceReviews.csv")

df = load_data()

# Define functions to create visualizations

def display_demographic_data():
    st.write("This is a Women’s Clothing E-Commerce dataset revolving around the reviews written by customers.Its nine supportive features offer a great environment to parse out the text through its multiple dimensions.Because this is real commercial data, it has been anonymized, and references to the company in the review text and body have been replaced with retailer")
    st.write(df.describe())
    st.header("Age,Rating and Positive Feedbackcounts")
    fig = px.scatter_3d(df, x='Age', y='Rating', z='Positive Feedback Count', color='Rating', size='Positive Feedback Count',
                         hover_data=['Age', 'Rating', 'Positive Feedback Count'],
                         title='Age Distribution - Rating vs Positive Feedback Count')
    st.plotly_chart(fig)
    st.header("Positive Feedback Count Distribution")
    fig = px.scatter(df, x='Positive Feedback Count', y='Age', title='Distribution of Positive Feedback Count')
    st.plotly_chart(fig)

def Imageprocessing():
    st.header("ImageProcessing")
    page1=st.selectbox("GoTo",["Resize","GrayscaleConversion","ImageCropping","ImageRotation"])
    def resize():
        im=Image.open("image1.jpg")
        st.title("orginal")
        st.image(im)
        st.title("resized")
        im2=im.resize((40,40))
        st.image(im2)
    def gray():
        st.title("orginal")
        im=Image.open("image2.jpg")
        st.image(im)
        st.title("Grayscale")
        im3=ImageOps.grayscale(im)
        st.image(im3)


    def crop():
        st.title("orginal")
        im=Image.open("image3.jpg")
        st.image(im)
        st.title("cropped")
        box=(20,20,70,70)
        region=im.crop(box)
        st.image(region)


    def rotate():
        im=Image.open("image4.jpg")
        st.title("orginal")
        st.image(im)
        out=im.rotate(45)
        st.title("Rotated")
        st.image(out)



    if page1 == "Resize":
        resize()
    elif page1=="GrayscaleConversion":
        gray()
    elif page1=="ImageCropping":
        crop()
    elif page1=="ImageRotation":
        rotate()
def textprocessing():
    def preprocess_text(text):
        tokens = nltk.word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum()]
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        return tokens
    
    # Extract the first 20 reviews
    text1_reviews = df['Review Text'][:20]
    with open('Sam.txt', 'w', encoding='utf-8') as text1_file:
        for review in text1_reviews:
            text1_file.write(review + '\n')
    with open('Sam.txt', 'r', encoding='utf-8') as file:
        Text= file.read()
    tokens1 = set(preprocess_text(Text))
    st.write("Preprocessed Text")
    st.write(tokens1)
    data=pd.DataFrame(df)
    review_text=data["Review Text"]
    division_name=data["Division Name"]
    general=data[division_name=='General']
    text_review_General=general
    with open('Sam1.txt', 'w', encoding='utf-8') as text2_file:
        for review in text_review_General:
            text2_file.write(review + '\n')
    with open('Sam1.txt', 'r', encoding='utf-8') as file:
        Text2= file.read()
    token2=set(preprocess_text(Text2))
    st.write(token2)
    general_petite=data[division_name=='General Petite']
    text_review_3=general_petite
    with open('Sam2.txt', 'w', encoding='utf-8') as text3_file:
        for review in text_review_3:
            text3_file.write(review + '\n')
    with open('Sam2.txt', 'r', encoding='utf-8') as file:
        Text3= file.read()
    token3=set(preprocess_text(Text3))
    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union
    similarity_score = jaccard_similarity(token2, token3)
    st.write(f"Jaccard Similarity: {similarity_score}")
    st.write(f"Tokens 1: {tokens1}\nTokens 2: {token2}")
    vectorizer = TfidfVectorizer()
    vector1 = vectorizer.fit_transform([' '.join(tokens1)])
    vector2 = vectorizer.transform([' '.join(token3)])
    cos_similarity = cosine_similarity(vector1, vector2)
    st.write(f"Cosine Similarity:\n{cos_similarity}")









    

    





















def display_positive_feedback_distribution():
    st.header("Positive Feedback Count Distribution")
    fig = px.scatter(df, x='Positive Feedback Count', y='Age', title='Distribution of Positive Feedback Count')
    st.plotly_chart(fig)

# Main function to run the app
def main():
    st.title("Women's Clothing E-Commerce Dashboard")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Demographic Data", "Imageprocessing", "TextProcessing"])

    if page == "Demographic Data":
        display_demographic_data()
    elif page == "Imageprocessing":
        Imageprocessing()
    
    elif page == "TextProcessing":
        textprocessing()

if __name__ == "__main__":
    main()
