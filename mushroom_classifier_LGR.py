import logging
from google.colab import drive
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ipywidgets import interactive, widgets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log start of script
logging.info("Starting the mushroom classification script.")

# Mount Google Drive
drive.mount('/content/drive')

# Load the data
data = pd.read_csv("/content/drive/MyDrive/mushroomClassification/mushrooms_revised.csv")
logging.info("Data loaded successfully.")

# Create a dictionary to store the mappings
mappings = {}

# Preprocess the data
encoder = LabelEncoder()
for col in data.columns:
    encoder.fit(data[col])
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    mappings[col] = mapping
    data[col] = encoder.transform(data[col])
logging.info("Data preprocessing completed.")

# Split the data into features and target
X = data.drop('class', axis=1)
y = data['class']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
logging.info("Data scaling completed.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
logging.info("Model training completed.")

# Evaluate the model's performance
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy*100:.2f}%")
logging.info(f"Model accuracy: {accuracy*100:.2f}%")

# Create a function to take user input and make predictions
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
    if prediction == 0:
        logging.info("The mushroom is predicted to be edible.")
        print("The mushroom is predicted to be edible.")
    else:
        logging.info("The mushroom is predicted to be poisonous.")
        print("The mushroom is predicted to be poisonous.")

# Create interactive widgets
cap_shape_widget = widgets.Dropdown(options=mappings['cap-shape'].keys())
cap_surface_widget = widgets.Dropdown(options=mappings['cap-surface'].keys())
cap_color_widget = widgets.Dropdown(options=mappings['cap-color'].keys())
bruises_widget = widgets.Dropdown(options=mappings['bruises'].keys())
odor_widget = widgets.Dropdown(options=mappings['odor'].keys())
gill_attachment_widget = widgets.Dropdown(options=mappings['gill-attachment'].keys())
gill_spacing_widget = widgets.Dropdown(options=mappings['gill-spacing'].keys())
gill_size_widget = widgets.Dropdown(options=mappings['gill-size'].keys())
gill_color_widget = widgets.Dropdown(options=mappings['gill-color'].keys())
stalk_shape_widget = widgets.Dropdown(options=mappings['stalk-shape'].keys())
stalk_root_widget = widgets.Dropdown(options=mappings['stalk-root'].keys())
stalk_surface_above_ring_widget = widgets.Dropdown(options=mappings['stalk-surface-above-ring'].keys())
stalk_surface_below_ring_widget = widgets.Dropdown(options=mappings['stalk-surface-below-ring'].keys())
stalk_color_above_ring_widget = widgets.Dropdown(options=mappings['stalk-color-above-ring'].keys())
stalk_color_below_ring_widget = widgets.Dropdown(options=mappings['stalk-color-below-ring'].keys())
veil_color_widget = widgets.Dropdown(options=mappings['veil-color'].keys())
ring_number_widget = widgets.Dropdown(options=mappings['ring-number'].keys())
ring_type_widget = widgets.Dropdown(options=mappings['ring-type'].keys())
spore_print_color_widget = widgets.Dropdown(options=mappings['spore-print-color'].keys())
population_widget = widgets.Dropdown(options=mappings['population'].keys())
habitat_widget = widgets.Dropdown(options=mappings['habitat'].keys())

interactive(predict_mushroom,
            cap_shape=cap_shape_widget,
            cap_surface=cap_surface_widget,
            cap_color=cap_color_widget,
            bruises=bruises_widget,
            odor=odor_widget,
            gill_attachment=gill_attachment_widget,
            gill_spacing=gill_spacing_widget,
            gill_size=gill_size_widget,
            gill_color=gill_color_widget,
            stalk_shape=stalk_shape_widget,
            stalk_root=stalk_root_widget,
            stalk_surface_above_ring=stalk_surface_above_ring_widget,
            stalk_surface_below_ring=stalk_surface_below_ring_widget,
            stalk_color_above_ring=stalk_color_above_ring_widget,
            stalk_color_below_ring=stalk_color_below_ring_widget,
            veil_color=veil_color_widget,
            ring_number=ring_number_widget,
            ring_type=ring_type_widget,
            spore_print_color=spore_print_color_widget,
            population=population_widget,
            habitat=habitat_widget)


import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('mappings.pkl', 'wb') as f:
    pickle.dump(mappings, f)
mushroom_classifier_LGR.ipynb
