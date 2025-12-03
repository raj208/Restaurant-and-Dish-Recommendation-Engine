import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sparse

# ==========================================
# 1. LOAD DATA & MODELS
# ==========================================
@st.cache_resource
def load_data():
    # Load CSVs
    interactions = pd.read_csv('data/interactions.csv')
    restaurants = pd.read_csv('data/restaurants.csv')
    
    # Load Models
    with open('data/als_model.pkl', 'rb') as f:
        als_data = pickle.load(f)
        
    with open('data/content_model.pkl', 'rb') as f:
        content_data = pickle.load(f)
        
    return interactions, restaurants, als_data, content_data

# Load everything once
try:
    df_interactions, df_restaurants, als_data, content_data = load_data()
except FileNotFoundError:
    st.error("Error: Files not found. Make sure 'interactions.csv', 'als_model.pkl', etc. are inside the 'data' folder.")
    st.stop()

# Extract model components
als_model = als_data['model']
user_mapper = als_data['user_mapper']
item_inv_mapper = als_data['item_inv_mapper']
item_mapper = als_data['item_mapper']

content_sim_matrix = content_data['similarity_matrix']
content_id_to_index = content_data['id_to_index']

# Create a sparse matrix for user history (needed for ALS)
unique_users = df_interactions['user_id'].unique()
unique_items = df_interactions['restaurant_id'].unique()
user_indices = df_interactions['user_id'].map(user_mapper).values
item_indices = df_interactions['restaurant_id'].map(item_mapper).values
confidence = df_interactions['strength'].values
sparse_user_item = sparse.csr_matrix((confidence, (user_indices, item_indices)), shape=(len(unique_users), len(unique_items)))

# ==========================================
# 2. RECOMMENDATION FUNCTIONS
# ==========================================
def get_als_recs(user_id, n=10):
    if user_id not in user_mapper:
        return []
    
    user_idx = user_mapper[user_id]
    
    # Get recommendations (indices, scores)
    ids, scores = als_model.recommend(user_idx, sparse_user_item[user_idx], N=n, filter_already_liked_items=True)
    
    recs = []
    for i, score in zip(ids, scores):
        # Clean check to avoid index errors
        if i in item_inv_mapper:
            recs.append({'restaurant_id': item_inv_mapper[i], 'als_score': float(score)})
    return recs

def get_content_recs(last_liked_item_id, n=10):
    if last_liked_item_id not in content_id_to_index:
        return []
    
    idx = content_id_to_index[last_liked_item_id]
    sim_scores = list(enumerate(content_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Skip the first one (it is the item itself)
    recs = []
    for i, score in sim_scores[1:n+1]:
        rest_id = df_restaurants.iloc[i]['restaurant_id']
        recs.append({'restaurant_id': rest_id, 'content_score': float(score)})
    return recs

def get_hybrid_recs(user_id):
    # 1. Get User's Last Liked Item (for Content-Based)
    user_history = df_interactions[df_interactions['user_id'] == user_id]
    if user_history.empty:
        return []
    last_item = user_history.iloc[-1]['restaurant_id']
    
    # 2. Get Candidates
    als_candidates = get_als_recs(user_id, n=20)
    content_candidates = get_content_recs(last_item, n=20)
    
    # 3. Merge & Score
    scores = {}
    
    # Normalize ALS scores (simple max division)
    max_als = max([x['als_score'] for x in als_candidates]) if als_candidates else 1
    
    for r in als_candidates:
        rid = r['restaurant_id']
        scores[rid] = scores.get(rid, {'als': 0, 'content': 0})
        scores[rid]['als'] = r['als_score'] / max_als
        
    for r in content_candidates:
        rid = r['restaurant_id']
        scores[rid] = scores.get(rid, {'als': 0, 'content': 0})
        scores[rid]['content'] = r['content_score']
        
    # Calculate Final Weighted Score
    final_recs = []
    for rid, s in scores.items():
        # HYBRID FORMULA: 50% Behavior + 50% Similarity
        final_score = (0.5 * s['als']) + (0.5 * s['content'])
        final_recs.append({'restaurant_id': rid, 'score': final_score, 'debug': s})
        
    final_recs = sorted(final_recs, key=lambda x: x['score'], reverse=True)[:10]
    return final_recs

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Swiggy AI Recommender", layout="wide")

st.title("🍔 Swiggy-Style Recommendation Engine")
st.markdown("A Hybrid System using **Collaborative Filtering (ALS)** and **Content-Based Filtering**.")

# Sidebar: User Selection
user_ids = sorted(df_interactions['user_id'].unique())
selected_user = st.sidebar.selectbox("Select User ID (Simulated)", user_ids)

# Show User History
st.sidebar.subheader("User History (Last 3)")
history = df_interactions[df_interactions['user_id'] == selected_user].tail(3)
for _, row in history.iterrows():
    r_name = df_restaurants[df_restaurants['restaurant_id'] == row['restaurant_id']]['name'].values[0]
    st.sidebar.write(f"- {r_name} ({row['interaction_type']})")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🔹 Collaborative Filtering", "🔸 Content-Based", "🚀 Hybrid Engine"])

with tab1:
    st.subheader(f"Top Picks for User {selected_user} (Behavior Only)")
    st.caption("Based on what similar users ordered.")
    
    als_recs = get_als_recs(selected_user)
    if als_recs:
        for rec in als_recs:
            r_info = df_restaurants[df_restaurants['restaurant_id'] == rec['restaurant_id']].iloc[0]
            st.info(f"**{r_info['name']}** | {r_info['cuisines']} | ₹{r_info['avg_cost_for_two']}")
    else:
        st.warning("No data found.")

with tab2:
    st.subheader("Because you liked...")
    user_hist = df_interactions[df_interactions['user_id'] == selected_user]
    
    if not user_hist.empty:
        last_item_id = user_hist.iloc[-1]['restaurant_id']
        last_item_name = df_restaurants[df_restaurants['restaurant_id'] == last_item_id]['name'].values[0]
        st.write(f"**{last_item_name}**")
        
        st.caption("Based on cuisine, price, and rating similarity.")
        
        content_recs = get_content_recs(last_item_id)
        if content_recs:
            for rec in content_recs:
                r_info = df_restaurants[df_restaurants['restaurant_id'] == rec['restaurant_id']].iloc[0]
                st.success(f"**{r_info['name']}** | {r_info['cuisines']} | Similarity: {rec['content_score']:.2f}")
    else:
        st.warning("User has no history.")

with tab3:
    st.subheader("🏆 Hybrid Recommendations")
    st.caption("Combining behavior (ALS) and relevance (Content).")
    
    hybrid_recs = get_hybrid_recs(selected_user)
    if hybrid_recs:
        for rec in hybrid_recs:
            r_info = df_restaurants[df_restaurants['restaurant_id'] == rec['restaurant_id']].iloc[0]
            
            # Visualizing the score contribution
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"### {r_info['name']}")
                st.text(f"{r_info['cuisines']} | ₹{r_info['avg_cost_for_two']}")
            with c2:
                st.metric("Hybrid Score", f"{rec['score']:.2f}")
                st.caption(f"ALS: {rec['debug']['als']:.2f} | Content: {rec['debug']['content']:.2f}")
            st.divider()
    else:
        st.write("No hybrid recommendations available.")