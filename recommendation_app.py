import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime

# Load model và dữ liệu đồng bộ
# ================================
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

# Mock user database 
USER_DB = {
    "stellarity": {"password": "12345678", "name": "Stellar"},
    "sheryluv": {"password": "luv123", "name": "Sheryl"}
}

def authenticate(username, password):
    return USER_DB.get(username, {}).get("password") == password

# Session state management 
if 'logged_in' not in st.sesion_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.view_history = []

# Login/Logout UI
def show_auth():
    with st.sidebar:
        if st.session_state.logged_in:
            st.success(f"Hello {USER_DB[st.session_state.user_name]['name']}!")
            if st.button('Sign out'):
                st.session_state.logged_in = False
                st.session_state.username = None 
                st.rerun()
        else:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in"):
                    if authenticate(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Username or password was wrong!!!")


def content_based_tab():
    st.markdown("##### 🛍️ Product Recommendation")
    # Search section
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input("Search for product: ", key="product_search")
    with col2:
        top_k = st.number_input("Number of product", min_value=1, max_value=20)

    # Filters
    with st.expander("🔍 Advanced Filtering"):
        price_range = st.slider("Price Range (VND)",
                                min_value=0,
                                max_value=int(df_product['price'].max()),
                                value=(0, int(df_product['price'].max())))
        min_rating = st.slider("Min ratings",
                                min_value=0.0,
                                max_value=5.0,
                                value=3.0, 
                                step=0.5)

    if st.button("Search", key="content_search"):
        if not user_input.strip():
            st.warning("Please enter the product description")
        else:
            with st.spinner("Searching for suitable products..."):
                # Vectorize input
                user_input_vector = vectorizer.transform([user_input])

                # Calculate similarity
                sim_scores = cosine_similarity(user_input_vector, tfidf_matrix)[0]

                # Apply filters 
                filtered_indices = apply_filters(price_range, min_rating)
                filtered_scores = [(idx, score) for idx, score in enumerate(sim_scores)
                                                if idx in filtered_indices]

                # Get top recommendations
                recommendations = process_recommendations(filtered_scores, top_k)

                # Display results
                display_results(recommendations)

def apply_filters(price_range, min_rating):
    mask = ((df_product['price'] >= price_range[0]) &
            (df_product['price'] <= price_range[1]) &
            (df_product['rating'] >= min_rating))
    return df_product[mask].index.tolist()

def process_recommendations(scores, top_k):
    top_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [{
        'name': df_product.iloc[idx]['product_name'],
        'price': df_product.iloc[idx]['price'],
        'rating': df_product.iloc[idx]['rating'],
        'score': score,
        'image': df_product.iloc[idx]['image']
    } for idx, score in top_indices]

def display_results(recommendations):
    st.markdown(f"##### 🎁 {len(recommendations)} suitable products")
    for item in recommendations:
        with st.container():
            col1, col2 = st.columns([1, 3])

            # Display image or placeholder if it's not existed image
            if pd.notna(item['image']):
                col1.image(item['image'], width=150, use_column_width=False)
            else:
                col1.image("no_image.png", width=150)

            col2.markdown(f"""
                **{item['name']}**\n
                💰 **{item['price']:,.0f} VND**\n
                ⭐ **Rating:** {item['rating']}/5\n
                🏷️ **Subcategory:** {item['sub_category']}\n
                🔍 **Relevance:** {item['score']:.2%}\n
                 [Details]({df_product[df_product['product_id'] == item['product_id']]['link'].values[0]})
            """)
            st.divider()

def user_based_tab():
    st.header('👤 User Recommendations')

    if not st.session_state .logged_in:
        st.warning("Please sign in to use this feature")
        return

    # User selection 
    selected_user = st.selectbox("Select User", valid_user_ids, format_func=lambda x: f"User {x}")
    top_k = st.slider("Number of Recommendations", 1, 20, 5, key="user_rec_slider")

    # Add filterings for user-based
    min_price, max_price = st.slider("Filter by Price", 
                                    min_value=0,
                                    max_value=int(df_product['price'].max()),
                                    value=(0, int(df_product['price'].max())))

    if st.button("Generate Recommendations", key="user_rec_btn"):
        with st.spinner("Analyzing user behavior..."):
            # Generate predictions
            predictions = generate_user_predictions(selected_user, top_k, min_price, max_price)

            # Display results
            display_user_results(predictions)
        
def generate_user_predictions(user_id, top_k, min_price, max_price):
    known_products = set(df_user[df_user['user_id'] == user_id]['product_id'])

    # Filtering product based on price range before making predictions
    valid_products = df_product[(df_product['price'] >= min_price) &
                                (df_product['price'] <= max_price)]['product_id']

    unknown_products = [pid for pid in valid_products if pid not in known_products]

    predictions = []
    for pid in unknown_products:
        pred = algo.predict(user_id, pid)
        predictions.append(pid, pred.est)

    return sorted(predictions, key=lambda x:x[1], reserve=True)[:top_k]

def display_user_results(predictions):
    st.markdown("##### 🔥 Recommended Products")
    for pid, rating in predictions:
        product = df_product[df_product['product_id'] == pid].iloc[0]

        with st.container():
            cols = st.columns([1, 3, 1])

            # Display image 
            if pd.notna(product['image']):
                cols[0].image(product['image'], width=120)
            else:
                cols[0].image("no_image.png", width=120)

            cols[1].markdown(f"""
                **{product['product_name']}**  
                💰 **{product['price']:,.0f} VND**  
                ⭐ **Predicted ratings:** {rating:.1f}/5  
                🏷️ **Subcategory:** {product['sub_category']}  
                [Details]({product['link']})
            """)

            if cols[2].button("💖", key=f"fav_{pid}"):
                st.toast(f"Saved {product['product_name']} to favorites")

            st.divider()

def show_history():
    if st.session_state.logged_in and st.session_state.view_history:
        with st.sidebar.expander("📚 Lịch sử hoạt động"):
            for item in reversed(st.session_state.view_history[-5:]):
                st.caption(f"{item['time'].strftime('%d/%m %H:%M')}")
                if item['type'] == "product_search":
                    st.write(f"🔍 Tìm kiếm: '{item['query']}'")
                else:
                    st.write(f"👥 Gợi ý cho User {item['target_user']}")
                
                for rec in item['results']:
                    st.image(rec['image'], width=60)
                st.divider()

def main():
    st.set_page_config(page_title="Recommendation System", layout="wide")
    show_auth()

    tab1, tab2 = st.tabs(["🛍️ Content-based filtering", "👥 Collaborative filtering"])

    with tab1:
        content_based_tab()
    
    with tab2:
        user_based_tab()

if __name__ == "__main__":
    main()