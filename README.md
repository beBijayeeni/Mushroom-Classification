# Mushroom Classification

This project aims to classify mushrooms as edible or poisonous using various machine learning models, with a focus on Logistic Regression.

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Results](#results)
- [User Interface](#user-interface)
- [Contributing](#contributing)
- [License](#license)
- [Additional Resources](#additional-resources)

## Project Description
The goal of this project is to predict which mushrooms are poisonous and which are edible based on their physical characteristics. The dataset used in this project is from the Audubon Society Field Guide to North American Mushrooms, which contains descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom (1981).

### Technologies
- Machine Learning
- Python
- Google Colab Notebooks

### Problem Statement
The main goal is to predict which mushroom is poisonous and which is edible based on given features.

## Installation

### Clone the Repository
```sh
git clone https://github.com/beBijayeeni/mushroom-classification.git
cd mushroom-classification
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Download the Dataset
Download the dataset from the provided link.

[Mushroom Classification Dataset](https://www.kaggle.com/uciml/mushroom-classification)

## Project Structure

```sh
mushroom-classification/
│
├── data/                    # Dataset
│   └── mushrooms_revised.csv
├── notebooks/               # Colab notebooks for exploration and model building
│   ├── mushroom_classification_EDA_model_comparison.ipynb
│   └── mushroom_classifier_LGR.ipynb
├── app/                     # Streamlit application
│   └── mushroom_classifier_app.py
├── tests/                   # Test cases for the project
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
```

## Results

### Model Performance
Summary of model performance, including accuracy scores and other relevant metrics.

### Random Forest Classifier:
- Accuracy: 100.00%
- Precision: 1.00
- Recall: 1.00
- F1 Score: 1.00

### Logistic Regression
- Accuracy: 96.00%
- Precision: 0.96
- Recall: 0.96
- F1 Score: 0.96

### Decision Tree
- Accuracy: 100.00%
- Precision: 1.00
- Recall: 1.00
- F1 Score: 1.00

### Support Vector Machine
- Accuracy: 99.94%
- Precision: 1.00
- Recall: 1.00
- F1 Score: 1.00

### Gradient Boosting
- Accuracy: 100.00%
- Precision: 1.00
- Recall: 1.00
- F1 Score: 1.00

### K-Nearest Neighbors
- Accuracy: 100.00%
- Precision: 1.00
- Recall: 1.00
- F1 Score: 1.00

### Model Selection
Considering the balance between accuracy, interpretability, and computational complexity, Logistic Regression is a preferable choice for this binary classification problem.

## User Interface

Web Application URL:

[Mushrrom Classification App](https://mushroom-classification-webapp.streamlit.app/)

![image](https://github.com/user-attachments/assets/b1041266-9d29-42e9-9e56-c60db74eedf0)

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

### Steps to Contribute
- Fork the repository.
- Create a new branch (`git checkout -b feature-branch`).
- Make your changes and commit them (`git commit -m 'Add some feature'`).
- Push to the branch (`git push origin feature-branch`).
- Open a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

### Additional Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Google Colab Documentation](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)
- [Python Documentation](https://docs.python.org/3/)
- [PEP 8 Coding Standards](https://peps.python.org/pep-0008/)
