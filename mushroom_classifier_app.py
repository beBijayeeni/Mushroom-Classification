import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
        background: linear-gradient(135deg, #71b7e6, #9b59b6);
        background-attachment: fixed;
    }
    .title {
        color: white;
        font-size: 3em;
        font-weight: bold;
    }
    .header, .subheader, .label {
        color: white;
    }
    .markdown-text-container {
        color: white;
    }
    .markdown-text-container p {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("Mushroom Classification 🍄 ")
st.write("Determine whether a mushroom is edible or poisonous based on its attributes.")

# Define function to make predictions
def predict_mushroom(cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size, gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_color, ring_number, ring_type, spore_print_color, population, habitat):
    input_data = pd.DataFrame({
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
    })
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)[0]
    return "Edible" if prediction == 0 else "Poisonous"

# Create the form for user input
with st.sidebar:
    st.header("Mushroom Attributes")
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
    result = predict_mushroom(cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size, gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_color, ring_number, ring_type, spore_print_color, population, habitat)
    if result == "Edible":
        st.success(f"The mushroom is predicted to be: **{result}**")
    else:
        st.error(f"The mushroom is predicted to be: **{result}**")

# Add a footer with more information
st.markdown("---")
st.markdown("This application uses a machine learning model- **logistic regression** to predict whether a mushroom is edible or poisonous based on its physical attributes. Always exercise caution and consult an expert before consuming wild mushrooms.")
