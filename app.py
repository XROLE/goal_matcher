import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize goal list
if "user_goals" not in st.session_state:
    st.session_state.user_goals = []

# Title
st.title("🤝 ProCircle Goal Matcher")
st.write("Submit your goal below. We'll pair you with someone who has a similar goal!")

# Form to collect goal input
with st.form("goal_form"):
    name = st.text_input("Your Name")
    goal = st.text_area("Your Weekly Goal")
    submitted = st.form_submit_button("Submit Goal")

    if submitted and name.strip() and goal.strip():
        st.session_state.user_goals.append({"name": name, "goal": goal})
        st.success("✅ Goal submitted successfully!")

# Display submitted goals
if st.session_state.user_goals:
    st.subheader("Submitted Goals")
    df = pd.DataFrame(st.session_state.user_goals)
    st.dataframe(df)

    # Match button
    if st.button("🔍 Match Participants"):
        goal_texts = [user["goal"] for user in st.session_state.user_goals]
        embeddings = model.encode(goal_texts)
        sim_matrix = cosine_similarity(embeddings)

        paired_indices = set()
        pairs = []

        for _ in range(len(goal_texts) // 2):
            max_sim = -1
            pair = (-1, -1)
            for i in range(len(goal_texts)):
                if i in paired_indices:
                    continue
                for j in range(i+1, len(goal_texts)):
                    if j in paired_indices:
                        continue
                    if sim_matrix[i][j] > max_sim:
                        max_sim = sim_matrix[i][j]
                        pair = (i, j)
            if pair != (-1, -1):
                paired_indices.update(pair)
                pairs.append((st.session_state.user_goals[pair[0]], st.session_state.user_goals[pair[1]]))

        # Display pairs
        st.subheader("👥 Matched Pairs")
        for u1, u2 in pairs:
            st.markdown(f"**{u1['name']}** (\"{u1['goal']}\")  🔗  **{u2['name']}** (\"{u2['goal']}\")")

        # Show unpaired
        unpaired = [user for idx, user in enumerate(st.session_state.user_goals) if idx not in paired_indices]
        if unpaired:
            st.warning(f"Unpaired: {unpaired[0]['name']} - \"{unpaired[0]['goal']}\"")
