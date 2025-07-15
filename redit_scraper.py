import streamlit as st
import praw
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline 
import os

nltk.download('punkt')
nltk.download('stopwords')

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

reddit = praw.Reddit(
    client_id='5OM6yVGDZWEE1TP3QnvuCQ',
    client_secret='OXnMebJeKtL_y9MKxDMKfUg_2YUYxg',
    user_agent = 'windows:MYbot:1.0 (by /u/Ok-Deer-8315)'
)

def get_reddit_user_data(username):
    user = reddit.redditor(username)
    comments, posts= [], []
    subreddits = defaultdict(int)

    for comment in user.comments.new(limit=None):
        comments.append(comment.body)
        subreddits[str(comment.subreddit)]+=1
    for post in user.submissions.new(limit=None):
        content= post.selftext if post.selftext else f"[{post.title}] {post.url}"
        posts.append(content)
        subreddits[str(post.subreddit)]+=1
        
    return comments, posts, subreddits

def build_user_persona(comments,posts):
    text= ' '.join(comments+posts)
    words = word_tokenize(text)
    stop_words= set(stopwords.words('english'))
    filtered= [w for w in words if w.lower() not in stop_words and w.isalpha()]
    return nltk.FreqDist(filtered)

# summarize
def generate_summary(text, word_freq, subreddits):
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summaries= [summarizer(chunk)[0]['summary_text'] for chunk in chunks[:3]]
    combined = ' '.join(summaries)
    return f"""
    Name:Infered from reddit activity
    Qualification: Possibly educated based on subreddits topics
    Goals: Interested in{', '.join(list(subreddits.keys())[:2])}
    Motivation: Engaged in learning/sharing via reddit
    Frustration: occasionaly critical or concerned
    Skills:shows strength in {', '.join(w for w, _ in word_freq.most_common(3))}
    Hobbies: Active in {', '.join(list(subreddits.keys())[:3])} 
    Summary of Reddit content (via LLM):{combined}
    """.strip()

# save to file 
def save_to_file(username,word_freq,subreddits, comments,posts,summary):
    filename=f"{username}_persona.txt"
    with open(filename, 'w', encoding='utf-8')as f:
        f.write(f"Reddit user persona: {username}\n\n")
        
        f.write("Most Active subreddits:\n")
        for sub, count in sorted(subreddits.items(),key=lambda x:x[1],reverse=True)[:5]:
            f.write(f" - {sub}: {count} posts/comments\n")
        f.write("\nTop 10 Most Used Words:\n")
        for word,freq in word_freq.most_common(10):
            f.write(f" - {word}: {freq}\n")
        f.write("\nSample Comments:\n")
        for c in comments[:3]:
            f.write(f"- {c[:150]}...\n")
        f.write("\nSample Posts:\n")
        for p in  posts[:3]:
            f.write(f"- {p[:150]}...\n")
        f.write("\n LLM-Based Persona Summary:\n")
        f.write(summary)
    return filename 

st.set_page_config(page_title="Reddit User Persona Generator", layout="centered")
st.title("Reddit User Persona Generator")
st.markdown("Build persona generator based on reddit user's profile using NLP and LLM")
url=st.text_input("Enter Reddit Profile URL : ")
if st.button("Generate persona"):
    if not url or "/user/" not in url :
        st.warning("Please enter a valid Reddit profile url ")
    else:
        with st.spinner("Fetching Reddit data..."):
            try:
                username= url.strip('/').split('/')[-1]
                comments,posts,subreddits= get_reddit_user_data(username)
                if not comments and not posts:
                    st.error("No public data")
                else:
                    word_freq= build_user_persona(comments,posts)
                    full_text=' '.join(comments+posts)
                    summary= generate_summary(full_text,word_freq, subreddits)
                    st.success("Persona generated")
                    st.markdown("LLM persona Summary")
                    st.code(summary)
                    filename=save_to_file(username,word_freq,subreddits,comments,posts,summary)
                    with open(filename, "rb") as f:
                        st.download_button("Download persona file", f, file_name=filename)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                
        

    
                      
            
    
                
    
    

        

    
    


