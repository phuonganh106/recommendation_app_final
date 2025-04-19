import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Recommendation System", 
    layout="wide",
    page_icon="✨"
)

# ====== CẤU HÌNH SIDEBAR ======
with st.sidebar:
    st.markdown("""
    **🎯 Recommender System Project**
    
    *Made by:*
    
    👩‍💻 **Nguyễn Thị Mai Linh**
    
    👨‍💻 **Tô Nguyễn Phương Anh**
    
    *Instructed by:*
    
    👩‍🏫 **Nguyễn Khuất Thùy Phương**
    
    *April 2025*
    """)
    st.markdown("---")  # Đường phân cách


# Now import other libraries
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.image("Shopee_logo.jpg", width=1200)

# ======================
# 1. INITIALIZATION
# ======================

# Constants
DEFAULT_IMAGE = "no_image.png"

# Load models and data
@st.cache_data
def load_models():
    with open('product_cosine.pkl', 'rb') as f:
        product_model = pickle.load(f)
    
    with open('surprise_model.pkl', 'rb') as f:
        user_model = pickle.load(f)
    
    return product_model, user_model

product_model, user_model = load_models()
vectorizer = product_model['vectorizer']
tfidf_matrix = product_model['tfidf_matrix']
df_product = product_model['dataframe']
algo = user_model['model']
df_user = user_model['df_sample']

# Filter valid users
valid_user_ids = df_user[df_user['product_id'].isin(df_product['product_id'])]['user_id'].unique().tolist()

# User database
USER_DB = {
    "admin": {"password": "12345678", "name": "Admin"},
    "phuonganh": {"password": "panh1006", "name": "Phuong Anh"},
    "mailinh": {"password": "mailinh97", "name": "Mai Linh"},
    "user123": {"password": "demo123", "name": "User 1"},
    "user456": {"password": "demo456", "name": "User 2"}
}

def authenticate(username, password):
    """Authenticate user"""
    return USER_DB.get(username, {}).get("password") == password

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.view_history = []
    st.session_state.show_login_form = False

# ======================
# 2. UI COMPONENTS
# ======================

def show_auth_header():
    """Show login/logout header"""
    if st.session_state.logged_in:
        cols = st.columns([4, 1])
        cols[0].markdown(f"### 👋 Welcome, {USER_DB[st.session_state.username]['name']}!")
        if cols[1].button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

def show_login_form(placeholder=None):
    """Display login form"""
    container = placeholder if placeholder else st
    with container.form("login_form", clear_on_submit=True):
        st.markdown("##### 🔐 Please sign in")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.form_submit_button("Sign in"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.show_login_form = False
                st.rerun()
            else:
                st.error("Incorrect username or password")

# ======================
# 3. RECOMMENDATION TABS
# ======================

def content_based_tab():
    """Content-based recommendations tab"""
    st.markdown("## 🛍️ Product Recommendation")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input("Search for product:", placeholder="Describe what you're looking for...")
    with col2:
        top_k = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5)

    # Filters
    with st.expander("🔍 Advanced Filters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min Price (VND)", 
                                      min_value=0,
                                      max_value=int(df_product['price'].max()),
                                      value=0,
                                      step=10000)
        with col2:
            max_price = st.number_input("Max Price (VND)", 
                                      min_value=0,
                                      max_value=int(df_product['price'].max()),
                                      value=int(df_product['price'].max()),
                                      step=10000)
        
        if min_price > max_price:
            st.error("Minimum price cannot be greater than maximum price")
            st.stop()
        
        min_rating = st.slider("Minimum rating", 
                              min_value=0.0,
                              max_value=5.0,
                              value=3.5, 
                              step=0.1)

    # Generate recommendations
    if st.button("Find Recommendations", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a product description")
        else:
            with st.spinner("Finding the best matches..."):
                user_input_vector = vectorizer.transform([user_input])
                sim_scores = cosine_similarity(user_input_vector, tfidf_matrix)[0]
                
                filtered_indices = apply_filters(min_price, max_price, min_rating)
                filtered_scores = [(idx, score) for idx, score in enumerate(sim_scores) 
                                 if idx in filtered_indices]
                
                recommendations = process_recommendations(filtered_scores, top_k)
                display_results(recommendations)

def user_based_tab():
    """Collaborative filtering recommendations tab"""
    st.markdown("## 👤 Personalized Recommendations")
    
    # Check login status
    if not st.session_state.logged_in:
        with st.expander("🧪 Try these demo accounts", expanded=True):
            st.markdown("""
            ### Demo Accounts (For testing only)
            | Username | Password | Role |
            |----------|----------|------|
            | `user123` | `demo123` | Normal User |
            | `user456` | `demo456` | Normal User |
            """)
        if st.button("🔐 Sign in to unlock personalized recommendations", type="primary"):
            st.session_state.show_login_form = True
        
        if st.session_state.show_login_form:
            show_login_form()
        return
    
    # For logged in users
    st.success(f"Welcome back, {USER_DB[st.session_state.username]['name']}!")
    
    # User selection and filters
    selected_user = st.selectbox("Select user profile", 
                               valid_user_ids, 
                               format_func=lambda x: f"User {x}")
    
    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input("Min price (VND)",
                                  min_value=0,
                                  max_value=int(df_product['price'].max()),
                                  value=0)
    with col2:
        max_price = st.number_input("Max price (VND)",
                                  min_value=0,
                                  max_value=int(df_product['price'].max()),
                                  value=int(df_product['price'].max()))
    
    top_k = st.slider("Number of recommendations", 1, 20, 5)

    # Generate recommendations
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Analyzing your preferences..."):
            predictions = generate_user_predictions(selected_user, top_k, min_price, max_price)
            display_user_results(predictions)

# ======================
# 4. HELPER FUNCTIONS
# ======================

def apply_filters(min_price, max_price, min_rating):
    """Apply price and rating filters"""
    mask = (
        (df_product['price'] >= min_price) & 
        (df_product['price'] <= max_price) & 
        (df_product['rating'] >= min_rating)
    )
    return df_product[mask].index.tolist()

def process_recommendations(scores, top_k):
    """Process recommendation scores"""
    top_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [{
        'product_id': df_product.iloc[idx]['product_id'],
        'name': df_product.iloc[idx]['product_name'],
        'price': df_product.iloc[idx]['price'],
        'rating': df_product.iloc[idx]['rating'],
        'sub_category': df_product.iloc[idx]['sub_category'],
        'score': score,
        'image': df_product.iloc[idx]['image']
    } for idx, score in top_indices]

def display_results(recommendations):
    """Display content-based recommendations"""
    st.markdown(f"##### 🎁 Found {len(recommendations)} matching products")
    for item in recommendations:
        with st.container():
            cols = st.columns([1, 3])
            
            img = item['image'] if pd.notna(item['image']) else DEFAULT_IMAGE
            cols[0].image(img, width=150, use_container_width=False)
            
            cols[1].markdown(f"""
                **{item['name']}**  
                💰 **Price:** {item['price']:,.0f} VND  
                ⭐ **Rating:** {item['rating']}/5  
                🏷️ **Category:** {item['sub_category']}  
                🔥 **Match score:** {item['score']:.0%}  
                [View details]({df_product[df_product['product_id'] == item['product_id']]['link'].values[0]})
            """)
            st.divider()

def generate_user_predictions(user_id, top_k, min_price, max_price):
    """Generate user-based predictions"""
    known_products = set(df_user[df_user['user_id'] == user_id]['product_id'])
    valid_products = df_product[
        (df_product['price'] >= min_price) & 
        (df_product['price'] <= max_price)
    ]['product_id']
    
    unknown_products = [pid for pid in valid_products if pid not in known_products]
    
    predictions = []
    for pid in unknown_products:
        pred = algo.predict(user_id, pid)
        predictions.append((pid, pred.est))
    
    return sorted(predictions, key=lambda x: x[1], reverse=True)[:top_k]

def display_user_results(predictions):
    """Display user-based recommendations"""
    st.markdown("### 🔥 Recommended For You")
    for pid, rating in predictions:
        product = df_product[df_product['product_id'] == pid].iloc[0]
        
        with st.container():
            cols = st.columns([1, 3, 1])
            
            img = product['image'] if pd.notna(product['image']) else DEFAULT_IMAGE
            cols[0].image(img, width=120, use_container_width=False)
            
            cols[1].markdown(f"""
                **{product['product_name']}**  
                💰 **Price:** {product['price']:,.0f} VND  
                ⭐ **Predicted rating:** {rating:.1f}/5  
                🏷️ **Category:** {product['sub_category']}  
                [View details]({product['link']})
            """)
            
            if cols[2].button("❤️", key=f"fav_{pid}"):
                st.toast(f"Saved {product['product_name']} to favorites")
            
            st.divider()

# ======================
# 5. MAIN APP
# ======================

def main():    
    # Show auth header
    show_auth_header()
    
    # Create tabs
    tab1, tab2 = st.tabs([
        "🔍 Content-Based Recommendations", 
        "👤 Personalized Recommendations"
    ])
    
    with tab1:
        content_based_tab()
    
    with tab2:
        user_based_tab()

if __name__ == "__main__":
    main()
