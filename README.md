# ML-project-1
This project is a machine learning model designed to perform sentiment analysis on movie reviews from the IMDb dataset. The goal of the model is to classify reviews as either positive or negative based on their content, providing insights into the sentiment expressed in the text.
Dataset
The model is trained and evaluated on the IMDb Movie Reviews dataset, a widely-used benchmark for sentiment analysis. The dataset contains 50,000 movie reviews labeled as positive or negative, split equally between the two classes to ensure a balanced training process.

50,000 reviews: 25,000 for training and 25,000 for testing.
Binary Sentiment Labels: positive (1) or negative (0).
Model Architecture
The sentiment analysis model is built using a Logistic Regression classifier, a simple yet effective method for binary classification tasks. The model is trained using text features extracted from the movie reviews.

Key Components:
Text Preprocessing: The text data is preprocessed to remove noise and standardize input. This includes:

Lowercasing all text.
Removing special characters and punctuation.
Tokenization (breaking down text into individual words or tokens).
Removing stop words (common words that do not contribute to the sentiment, like "and", "the", etc.).
Feature Extraction: The model uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features. This technique helps highlight the most important words in the dataset by reducing the weight of commonly used words.

Model Training: The Logistic Regression model is trained on the transformed features, optimizing the parameters to distinguish between positive and negative reviews.
Performance
The model achieved an accuracy of 89.08% on the test set, indicating strong performance in predicting sentiment correctly. Other evaluation metrics include precision, recall, F1-score, and the confusion matrix, which provide a more comprehensive understanding of model performance.
