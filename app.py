import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import re
import random
import time
import hydralit_components as hc
import pandas as pd


# Set layout to wide
st.set_page_config(layout="wide")

# Create a function to analyze the dataset
def analyze_data(df):
    st.write("Analysis Results:")
    # Display some basic information about the data
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
    # Do analysis here and display results
    # ...

# Create functions to preprocess the dataset
handles_processed = []
ALLOW_NEW_LINES = False
EPOCHS = 4

def fix_text(text):
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        return text
    
def clean_tweet(tweet, allow_new_lines = ALLOW_NEW_LINES):
        bad_start = ['http:', 'https:']
        for w in bad_start:
            tweet = re.sub(f" {w}\\S+", "", tweet)      # removes white space before url
            tweet = re.sub(f"{w}\\S+ ", "", tweet)      # in case a tweet starts with a url
            tweet = re.sub(f"\n{w}\\S+ ", "", tweet)    # in case the url is on a new line
            tweet = re.sub(f"\n{w}\\S+", "", tweet)     # in case the url is alone on a new line
            tweet = re.sub(f"{w}\\S+", "", tweet)       # any other case?
        tweet = re.sub(' +', ' ', tweet)                # replace multiple spaces with one space
        if not allow_new_lines:                         # TODO: predictions seem better without new lines
            tweet = ' '.join(tweet.split())
        return tweet.strip()

def boring_tweet(tweet):
        "Check if this is a boring tweet"
        boring_stuff = ['http', '@', '#']
        not_boring_words = len([None for w in tweet.split() if all(bs not in w.lower() for bs in boring_stuff)])
        return not_boring_words < 3

def preprocess_data(df):
    st.write("Preprocessing Dataset:")
    # Do preprocessing here and display results
    res = {}
    res["tweets"] = df["Text"].tolist()
    res['n_tweets'] = len(res["tweets"])
    res['n_RT'] = df.Retweets.sum()
    all_tweets = res["tweets"]
    cool_tweets = []
    handles_processed = []
    raw_tweets = []
    user_names = []
    n_tweets_dl = []
    n_retweets = []
    n_short_tweets = []
    n_tweets_kept = []
    i = 0
    raw_tweets.append(all_tweets)
    curated_tweets = [fix_text(tweet) for tweet in all_tweets]
    # create dataset
    clean_tweets = [clean_tweet(tweet) for tweet in curated_tweets]
    cool_tweets.append([tweet for tweet in clean_tweets if not boring_tweet(tweet)])
    # save count
    n_tweets_dl.append(str(res['n_tweets']))
    n_retweets.append(str(res['n_RT']))
    n_short_tweets.append(str(len(all_tweets) - len(cool_tweets[-1])))
    n_tweets_kept.append(str(len(cool_tweets[-1])))
    
    if len('<|endoftext|>'.join(cool_tweets[-1])) < 6000:
        # need about 4000 chars for one data sample (but depends on spaces, etc)
        raise ValueError(f"Error: this user does not have enough tweets to train a Neural Network\n{res['n_tweets']} tweets downloaded, including {res['n_RT']} RT's and {len(all_tweets) - len(cool_tweets)} boring tweets... only {len(cool_tweets)} tweets kept!")
    if len('<|endoftext|>'.join(cool_tweets[-1])) < 40000:
        st.write('\n<b>Warning: this user does not have many tweets which may impact the results of the Neural Network</b>\n')
    st.write(f"\n{n_tweets_dl[-1]} tweets detected, including {n_retweets[-1]} RT's and {n_short_tweets[-1]} short tweets... keeping {n_tweets_kept[-1]} tweets\n\n\n")
    # create a file based on multiple epochs with tweets mixed up
    seed_data = random.randint(0,2**32-1)
    dataRandom = random.Random(seed_data)
    total_text = '<|endoftext|>'
    all_handle_tweets = []
    epoch_len = max(len(''.join(cool_tweet)) for cool_tweet in cool_tweets)
    for _ in range(EPOCHS):
        for cool_tweet in cool_tweets:
            dataRandom.shuffle(cool_tweet)
            current_tweet = cool_tweet
            current_len = len(''.join(current_tweet))
            while current_len < epoch_len:
                for t in cool_tweet:
                    current_tweet.append(t)
                    current_len += len(t)
                    if current_len >= epoch_len: break
            dataRandom.shuffle(current_tweet)
            all_handle_tweets.extend(current_tweet)
            total_text += '<|endoftext|>'.join(all_handle_tweets) + '<|endoftext|>'

    st.write('\nCreating dataset...')
    # for 4 replications of the same loader (index=2) from the standard loader group
    with hc.HyLoader('Processing',hc.Loaders.standard_loaders,index=0):
        time.sleep(8)
    #if st.button("Save Processed Dataset"):
    with open(f"data_{df['Screen Name'][0]}_train.txt", 'w', encoding='utf-8') as f:
        f.write(total_text)
    st.write('\n🎉 Dataset created successfully!')
    st.write('\n🎉 Dataset saved successfully!')
    
    
def long_running_process():
    for i in range(10):
        time.sleep(0.5)
        st.experimental_rerun()
        
# Set up the Streamlit app
st.title("Twitter Data Analysis and Preprocessing App")
st.write("This app allows you to analyze Twitter data from an Excel file and preprocess tweets dataset for Fine Tuning Large Language Models.")
st.write("Upload a dataset in CSV format:")
st.sidebar.title("Features")
# Allow user to upload a dataset
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# If a dataset has been uploaded, display a button to analyze it
if uploaded_file is not None:
    # Read the dataset into a Pandas dataframe
    df = pd.read_excel(uploaded_file)
    # Display the first few rows of the dataset
    if st.button("View Dataset"):
        st.write("Dataset Preview:")
        st.write(df.head())

    # Display buttons for analysis and preprocessing
    if st.sidebar.button("Analyze Data"):
        analyze_data(df)

    if st.sidebar.button("Preprocess Data"):
        preprocess_data(df)
