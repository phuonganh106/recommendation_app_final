import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Project Overview", page_icon="💞")
st.title("💞 Welcome to Recommender System Project! 💞")
st.markdown("---")

with open('product_cosine.pkl', 'rb') as f:
    product_model = pickle.load(f)

vectorizer = product_model['vectorizer']
tfidf_matrix = product_model['tfidf_matrix']
df_product = product_model['dataframe']

with open('surprise_model.pkl', 'rb') as f:
    user_model = pickle.load(f)

algo = user_model['model']
df_user = user_model['df_sample']

valid_user_ids = df_user[df_user['product_id'].isin(df_product['product_id'])]['user_id'].unique().tolist()

# SIDEBAR
with st.sidebar:
    # Phần tiêu đề
    st.markdown("""
    <h3 style='text-align: center; margin-bottom: 0.5em;'>🎯 Recommender System</h3>
    <p style='text-align: center; font-style: italic; margin-top: 0;'>Graduation Project</p>
    """, unsafe_allow_html=True)
    
    # Phần thông tin đồ án (dùng markdown đơn giản)
    st.markdown("""
    **📚 Course:**  
    DL07_DATN_K302_T37  
    
    **🏫 Institution:**  
    Trung Tâm Tin Học - Trường ĐH KHTN  
    
    **👩‍🏫 Instructor:**  
    Cô Nguyễn Khuất Thùy Phương  
    
    **👨‍💻 Authors:**  
    - Nguyễn Thị Mai Linh  
    - Tô Nguyễn Phương Anh  
    
    **📅 Date:**  
    April 2025
    """)
    
    # Đường phân cách cuối
    st.markdown("---")



# Tạo khoảng trắng (20px)
st.markdown(
    """
    <style>
        .custom-space {
            margin-top: 20px;
            margin-bottom: 20px;
        }
    </style>
    <div class="custom-space"></div>
    """,
    unsafe_allow_html=True)

# Project introduction
st.markdown("#### 📌 Project Introduction")
intro = """
  <div style="text-align: justify; padding: 10px; border-radius: 10px; background-color: #f5f5f5;">
  With over 10,000+ men's fashion products on Shopee, users often feel overwhelmed by choices. 
  Our system solves this by combining <b>collaborative filtering</b> and <b>content-based filtering</b> to provide personalized recommendation that match both user preferences and product characteristics.
  </div>
"""
st.markdown(intro, unsafe_allow_html=True)
st.markdown("")

# Project goals in columns
st.markdown("#### 🎯 Project Goals")
goals = st.columns(3)

with goals[0]:
  st.info("""
  **Personalization**:
  👕 Tailor recommendations to individual tastes
  """)

with goals[1]:
  st.success("""
  **Discovery**
  🔍 Help users find items they'll love but might miss
  """)

with goals[2]:
  st.warning("""
  **Engagement**
  ⏳ Reduce time spent searching
  """)

st.markdown("---")

# Data visualization section
st.markdown("#### 📊 Dataset Overview")

# Sample data metrics
data_cols = st.columns(4)
data_cols[0].metric("Products", "48K+")
data_cols[1].metric("Users", "650K+")
data_cols[2].metric("Ratings", "1M")
data_cols[3].metric("Sub Categories", "17")

# Wordcloud & Data distribution
tab1, tab2, tab3 = st.tabs(["Product Keywords", "Product's Ratings Distribution", "User's Ratings Distribution"])

with tab1:
  st.markdown("##### Wordcloud of Product Keywords")
  text = ' '.join(df_product['final_cleaned_tokens'].apply(lambda x: ' '.join(x)))
  wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
  fig, ax = plt.subplots(figsize=(10, 5))
  ax.imshow(wordcloud, interpolation='bilinear')
  ax.axis('off')
  st.pyplot(fig)

with tab2:
  st.markdown("##### Product's Ratings Distribution")
  st.bar_chart(df_product['rating'].value_counts().sort_index())

with tab3:
  st.markdown("##### User's Ratings Distribution")
  st.bar_chart(df_user['rating'].value_counts().sort_index())
st.markdown("---")

# Model explanation
st.markdown("#### 🤖 Recommendation Approaches")

approach = st.selectbox(
    "Select approach to learn more:",
    ["SVD++ (Collaborative Filtering)", "Cosine Similarity (Content-Based)"]
)

if "SVD++" in approach:
    st.markdown("""
    # **Matrix Factorization Technique**
    # - Decomposes user-item interaction matrix
    # - Captures latent factors in user preferences 
    # - Handles implicit feedback
    **📌 Matrix Factorization Technique (Collaborative Filtering)**
    - **What is Collaborative Filtering?**  
      A method that predicts user preferences based on user behavior (ratings, interactions) without requiring product features.
    - **How SVD++ works:**  
      - Decomposes user-item interaction matrix (ratings)  
      - Captures latent factors in user preferences and item characteristics  
      - Effectively handles implicit feedback (views, purchases)  
    - **Benefits:**  
      → Works well for cold-start users with some interaction history  
      → Discovers hidden patterns in user behavior 
    """)
else:
    st.markdown("""
    # **Content-Based Filtering**
    # - Analyzes product features (description, category, price)
    # - Use TF-IDF for text processing
    # - Calculates similarity between items
    **📌 Content-Based Filtering**
    - **What is Content-Based Filtering?**  
      Recommends items by analyzing product features and matching them to user profiles.
    - **How Cosine Similarity works:**  
      - Uses TF-IDF to process text data (descriptions, categories)  
      - Calculates similarity between items using cosine distance  
      - Focuses on product attributes (price, category, etc.)  
    - **Benefits:**  
      → Works without user interaction data  
      → Transparent recommendations (explainable by features) 
    """)
st.markdown("---")

# --- Data cleaning process ---
st.markdown("#### 🏗 🧼 Data Preprocessing Pipeline")

# Graphviz diagram
st.graphviz_chart("""
digraph {
    node [shape=rectangle, style=rounded, color="#EE4D2D", fontname="Helvetica", fontsize=12]
    rankdir = LR
    nodesep = 0.5

    "1. Text Normalization" [fillcolor="#FFE5D9", style="filled, rounded"]
    "2. Noise Removal" [fillcolor="#FFE5D9", style="filled, rounded"]
    "3. Tokenization" [fillcolor="#FFE5D9", style="filled, rounded"]
    "4. Final Output" [fillcolor="#D4EDDA", style="filled,rounded"]

    "1. Text Normalization" -> "2. Noise Removal"
    "2. Noise Removal" -> "3. Tokenization"
    "3. Tokenization" -> "4. Final Output"
    }

""")




