import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set layout to wide
st.set_page_config(layout="wide")

# Define the app description
st.write("""
# Twitter Data Analysis App

This app allows you to analyze Twitter data from an Excel file. This app will be further developed to preprocess this tweets dataset for Fine Tuning Large Language Models in future.

""")

st.subheader("Upload xlsx File of Tweets")
st.write("""Go to 'https://www.vicinitas.io/' and download the tweets of your desired Twitter handle by writing username of the twitter account and then an xlsx file will be downloaded. Upload that file below to Analyze Tweet Data.""")
# Define a function to allow the user to upload a file
def file_upload():
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df
    else:
        return None

# Allow the user to upload a file and store it in a variable
file = file_upload()

# If a file was uploaded, display its data using Streamlit
if file is not None:
    # Use pandas to read in the Excel file
    df = pd.read_excel('elonmusk_user_tweets.xlsx', usecols=['Tweet Id', 'Text', 'Name', 'Screen Name', 'UTC', 'Created At',
       'Favorites', 'Retweets', 'Language', 'Client', 'Tweet Type'])
    # Display some basic information about the data
    st.sidebar.write("## About")
    st.sidebar.write("This app allows you to analyze Twitter data from an Excel file.This app will be further developed to preprocess this tweets dataset for Fine Tuning Large Language Models in future.")
    st.sidebar.write(f"Number of tweets: {len(df)}")
    st.sidebar.write(f"Languages present: {df['Language'].value_counts().idxmax()}")
    Name = df["Name"][0]
    Username = df["Screen Name"][0]
    st.write(f"""### Name of Twitter User: {Name}""")
    st.write(f"""### Username of Twitter User: {Username}""")
    # Display the data in a table
    st.subheader("Dataset")
    st.dataframe(df)
    
    st.subheader("Select a Date to Visualize Tweets")
    st.write("You can see the time period of tweets in the sidebar to select appropriate date which has values stored in dataset.")
    # Convert created_at to datetime
    df["Created At"] = pd.to_datetime(df["Created At"], format='%a %b %d %H:%M:%S %z %Y')
    
    # Allow user to select a date
    selected_date = st.date_input("Select a date", min_value=df["Created At"].min().date(), max_value=df["Created At"].max().date(), value=df["Created At"].min().date())

    #selected_date = st.date_input("Select a date", df["Created At"].min(), df["Created At"].max())

    # Filter the data for the selected date
    selected_data = df[df["Created At"].dt.date == selected_date]

    # Display the tweets
    st.write(selected_data[["Created At", "Text"]])

    # Display 
    st.write("""# Basic Analysis""")
    st.subheader("Bar Chart of the Number of Tweets by Tweet Type")
    tweet_counts = df['Tweet Type'].value_counts()
    st.bar_chart(tweet_counts)

    # Display 
    st.subheader("Scatter Plot of Retweets vs Favorites")
    st.write("Retweets vs. Favorites")
    st.write("")

    scatter_data = df[['Retweets', 'Favorites']]
    st.line_chart(scatter_data)
    
    # Convert the 'Created At' column to a datetime format
    df['Created At'] = pd.to_datetime(df['Created At'], format='%a %b %d %H:%M:%S %z %Y')

    # Calculate the duration of tweets from when to when tweets were posted
    start_time = df['Created At'].min()
    end_time = df['Created At'].max()
    duration = end_time - start_time

    st.sidebar.write(f"Tweets were posted from {start_time} to {end_time}.")
    st.sidebar.write(f"The duration of tweets since posted was {duration}.")
    #st.dataframe(file)
    st.subheader("Wordcloud For Tweets")
    text = " ".join(tweet for tweet in df.Text)
    st.write("There are {} words in the combination of all tweets.".format(len(text)))
    # lower max_font_size, change the maximum number of word and lighten the background:
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)

