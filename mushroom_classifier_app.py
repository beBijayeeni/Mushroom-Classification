import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the model and scalers
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Mushroom Classification", page_icon="🍄", layout="centered")

# Inject custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://wallpapers.com/images/featured/mushroom-background-hazdm3w8w3nta7h2.jpg") no-repeat center center fixed;
        background-size: cover;
	 padding: 20px;
    }

    .title-text {
        color: #ffcc00 !important; 
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
	font-family: 'Arial', sans-serif;
    	font-size: 2.5em;
    	text-align: left;
    	margin-bottom: 10px; 
    }

    .description-text {
        color: #000000 !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5);
	font-family: 'Arial', sans-serif;
    	font-size: 1.2em;
    	text-align: left;
    	margin-bottom: 30px;
    }
    .stButton > button {
    background-color: #ff3300
    color: white;
    border: none;
    border-radius: 25px;
    padding: 25px 25px;
}

.stButton > button:hover {
    background-color: #ff6c4c; 
    color: white
}

@media only screen and (max-width: 600px) {
    .stSidebar {
        width: 100%;
        position: relative;
    }
}




    </style>
    """, unsafe_allow_html=True
)

st.markdown('<h1 class="title-text">Mushroom Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="description-text"><b>Determine whether a mushroom is edible or poisonous based on its attributes 🍽️⚠️</b></p>', unsafe_allow_html=True)


# Define function to make predictions
def predict_mushroom(attributes):
    try:
        input_data = pd.DataFrame(attributes)
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)[0]
        return "Edible" if prediction == 0 else "Poisonous"
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Create the form for user input
with st.sidebar:
    st.header("Mushroom Attributes 🍄")
    st.write("Please select the attributes of the mushroom:")

    cap_shape = st.selectbox("Cap Shape", options=list(mappings['cap-shape'].keys()))
    cap_surface = st.selectbox("Cap Surface", options=list(mappings['cap-surface'].keys()))
    cap_color = st.selectbox("Cap Color", options=list(mappings['cap-color'].keys()))
    bruises = st.selectbox("Bruises", options=list(mappings['bruises'].keys()))
    odor = st.selectbox("Odor", options=list(mappings['odor'].keys()))
    gill_attachment = st.selectbox("Gill Attachment", options=list(mappings['gill-attachment'].keys()))
    gill_spacing = st.selectbox("Gill Spacing", options=list(mappings['gill-spacing'].keys()))
    gill_size = st.selectbox("Gill Size", options=list(mappings['gill-size'].keys()))
    gill_color = st.selectbox("Gill Color", options=list(mappings['gill-color'].keys()))
    stalk_shape = st.selectbox("Stalk Shape", options=list(mappings['stalk-shape'].keys()))
    stalk_root = st.selectbox("Stalk Root", options=list(mappings['stalk-root'].keys()))
    stalk_surface_above_ring = st.selectbox("Stalk Surface Above Ring", options=list(mappings['stalk-surface-above-ring'].keys()))
    stalk_surface_below_ring = st.selectbox("Stalk Surface Below Ring", options=list(mappings['stalk-surface-below-ring'].keys()))
    stalk_color_above_ring = st.selectbox("Stalk Color Above Ring", options=list(mappings['stalk-color-above-ring'].keys()))
    stalk_color_below_ring = st.selectbox("Stalk Color Below Ring", options=list(mappings['stalk-color-below-ring'].keys()))
    veil_color = st.selectbox("Veil Color", options=list(mappings['veil-color'].keys()))
    ring_number = st.selectbox("Ring Number", options=list(mappings['ring-number'].keys()))
    ring_type = st.selectbox("Ring Type", options=list(mappings['ring-type'].keys()))
    spore_print_color = st.selectbox("Spore Print Color", options=list(mappings['spore-print-color'].keys()))
    population = st.selectbox("Population", options=list(mappings['population'].keys()))
    habitat = st.selectbox("Habitat", options=list(mappings['habitat'].keys()))
if st.button("Predict"):
    attributes = {
        'cap-shape': [mappings['cap-shape'][cap_shape]],
        'cap-surface': [mappings['cap-surface'][cap_surface]],
        'cap-color': [mappings['cap-color'][cap_color]],
        'bruises': [mappings['bruises'][bruises]],
        'odor': [mappings['odor'][odor]],
        'gill-attachment': [mappings['gill-attachment'][gill_attachment]],
        'gill-spacing': [mappings['gill-spacing'][gill_spacing]],
        'gill-size': [mappings['gill-size'][gill_size]],
        'gill-color': [mappings['gill-color'][gill_color]],
        'stalk-shape': [mappings['stalk-shape'][stalk_shape]],
        'stalk-root': [mappings['stalk-root'][stalk_root]],
        'stalk-surface-above-ring': [mappings['stalk-surface-above-ring'][stalk_surface_above_ring]],
        'stalk-surface-below-ring': [mappings['stalk-surface-below-ring'][stalk_surface_below_ring]],
        'stalk-color-above-ring': [mappings['stalk-color-above-ring'][stalk_color_above_ring]],
        'stalk-color-below-ring': [mappings['stalk-color-below-ring'][stalk_color_below_ring]],
        'veil-color': [mappings['veil-color'][veil_color]],
        'ring-number': [mappings['ring-number'][ring_number]],
        'ring-type': [mappings['ring-type'][ring_type]],
        'spore-print-color': [mappings['spore-print-color'][spore_print_color]],
        'population': [mappings['population'][population]],
        'habitat': [mappings['habitat'][habitat]]
    }
    result = predict_mushroom(attributes)
    if result:
        st.write(f"The mushroom is predicted to be: **{result}**")

# Add a footer with more information and contact links
st.markdown("---")
st.markdown("This application uses a machine learning model - **Logistic Regression** - to predict whether a mushroom is edible or poisonous based on its physical attributes.")

# Add contact links with icons
st.markdown(
    """
    <footer>
	<br>
	<br>
	<br>
        <p><b>Connect with me:</b></p>
        <a href="https://www.linkedin.com/in/bijayeeni-halder-0b1037289/" target="_blank">
            <img src="https://img.icons8.com/material-outlined/24/000000/linkedin.png" alt="LinkedIn" style="height: 40px; width: 40px; margin-right: 10px;">
        </a>
        <a href="https://github.com/beBijayeeni" target="_blank">
            <img src="https://img.icons8.com/material-outlined/24/000000/github.png" alt="GitHub" style="height: 40px; width: 40px; margin-right: 10px;">
        </a>
    </footer>
    """, unsafe_allow_html=True
)
