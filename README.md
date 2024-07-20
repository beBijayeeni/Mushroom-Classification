Mushroom Classification
This project aims to classify mushrooms as edible or poisonous using various machine learning models, with a focus on Logistic Regression.

Table of Contents
Project Description
Installation
Usage
Project Structure
Results
Contributing
License
Project Description
The goal of this project is to predict which mushrooms are poisonous and which are edible based on their physical characteristics. The dataset used in this project is from the Audubon Society Field Guide to North American Mushrooms, which contains descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom (1981).

Technologies
Machine Learning
Python
Flask (for API)
Jupyter Notebooks
Problem Statement
The main goal is to predict which mushroom is poisonous and which is edible based on given features.

Installation
Clone the Repository
sh
Copy code
git clone https://github.com/yourusername/mushroom-classification.git
cd mushroom-classification
Create a Virtual Environment and Activate it
sh
Copy code
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
Install Dependencies
sh
Copy code
pip install -r requirements.txt
Download the Dataset
Download the dataset from the provided link and place it in the data/ directory.

Dataset: Mushroom Classification Dataset (or use your dataset path)

Usage
Jupyter Notebooks
Explore the data and build the model using the Jupyter notebooks provided in the notebooks/ directory.

sh
Copy code
jupyter notebook notebooks/notebook1.ipynb
jupyter notebook notebooks/notebook2.ipynb
Running the API
Ensure you have Flask installed and the model is trained.
Run the Flask app.
sh
Copy code
python src/app.py
Use the API endpoint to get predictions.
sh
Copy code
curl -X POST -H "Content-Type: application/json" -d '{"cap_shape": "x", "cap_surface": "s", "cap_color": "n", "bruises": "t", "odor": "p", "gill_attachment": "f", "gill_spacing": "c", "gill_size": "b", "gill_color": "k", "stalk_shape": "e", "stalk_root": "e", "stalk_surface_above_ring": "s", "stalk_surface_below_ring": "s", "stalk_color_above_ring": "w", "stalk_color_below_ring": "w", "veil_color": "w", "ring_number": "o", "ring_type": "p", "spore_print_color": "k", "population": "s", "habitat": "u"}' http://127.0.0.1:5000/predict
Project Structure
bash
Copy code
mushroom-classification/
│
├── data/                    # Dataset and related files
│   └── mushrooms_revised.csv
├── notebooks/               # Jupyter notebooks for exploration and model building
│   ├── notebook1.ipynb
│   └── notebook2.ipynb
├── src/                     # Source code for the project
│   ├── app.py               # Flask API
│   ├── data_preprocessing.py# Data preprocessing scripts
│   ├── model.py             # Model building and training scripts
│   └── utils.py             # Utility functions
├── tests/                   # Test cases for the project
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
Results
Model Performance
Summary of model performance, including accuracy scores and other relevant metrics.

Random Forest Classifier:

Accuracy: 100.00%
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
Logistic Regression:

Accuracy: 94.31%
Precision: 0.94
Recall: 0.94
F1 Score: 0.94
Decision Tree Classifier:

Accuracy: 100.00%
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
Support Vector Machine:

Accuracy: 100.00%
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
Gradient Boosting Classifier:

Accuracy: 100.00%
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
K-Nearest Neighbors:

Accuracy: 100.00%
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
Analysis
Logistic Regression performed well with a 94.31% accuracy, but models like Random Forest, Decision Tree, SVM, Gradient Boosting, and K-Nearest Neighbors achieved perfect accuracy.

Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

Steps to Contribute:
Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit them (git commit -m 'Add some feature').
Push to the branch (git push origin feature-branch).
Open a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Additional Resources
PEP 8 - Style Guide for Python Code
Flask Documentation
Jupyter Notebooks Documentation
