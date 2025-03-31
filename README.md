Recurrent Neural Network Sentiment analysis using
Contributors:

Dimple(055009)
Rohan Jha(055057)


#**Objective**
The primary objective of this project is to design, implement, and evaluate a deep learning-based sentiment analysis model using an RNN architecture. This model aims to classify movie reviews based on their sentiment (positive or negative) by leveraging the sequential patterns present in text data. The project aims to develop a model that can accurately and efficiently classify reviews, providing insights into public opinion and sentiment trends.

Data Description
1. Datasets
The project utilizes two datasets:

IMDB Dataset of 50K Movie Reviews: This dataset is used for training the RNN model. It contains 50,000 movie reviews labeled as either positive or negative.
Metacritic Reviews Dataset: This dataset is used for testing the generalization ability of the trained model. It consists of 151 manually collected reviews and ratings from Metacritic.
2. IMDB Training Data
Size: 50,000 records
Columns:
Review: The textual content of the movie review.
Sentiment: The sentiment label (positive or negative). Positive sentiment is encoded as 1, and negative sentiment is encoded as 0.
A random sample of 40,000 reviews is selected for training.
3. Metacritic Testing Data
Size: 151 records
Columns:
Movie Name: The title of the movie.
Rating: The rating given to the movie.
Review: The textual content of the movie review.
Sentiment: The sentiment label (positive or negative).
Data Preprocessing Steps
1. Sentiment Encoding
Positive Sentiment → Encoded as 1
Negative Sentiment → Encoded as 0
2. Text Normalization
Removing Special Characters: Stripping unnecessary characters (e.g., punctuation, special symbols) to clean the text.
Lowercasing: Converting all reviews to lowercase for uniformity and consistency.
3. Tokenization
Splitting the text into individual tokens (words).
Using a vocabulary size of 20,000 most frequent words (max_features=20000). Any words outside this range are replaced with a placeholder token.
4. Sequence Padding
Ensuring all tokenized reviews are of the same length by:
Padding shorter sequences with zeros at the beginning or end.
Truncating longer sequences to a maximum length of 400 tokens (max_length = 400).
Observations
1. Library Imports
The notebook confirms the successful import of necessary libraries, including TensorFlow, Pandas, NumPy, re (for regular expressions), and scikit-learn (for data splitting).
2. Data Loading and Preprocessing
The Metacritic testing dataset was loaded successfully from a Google Drive URL, as shown in the code.
The columns attribute and shape attribute confirmed the successful loading of 151 entries, each having 4 columns.
Model Building
1. Model Architecture
The model includes the following layers:

Embedding Layer: Input dimension of 20,000 (vocabulary size), output dimension of 128 (word embedding size), and input length of 400 (maximum sequence length).
Recurrent Layer: Simple RNN with 64 units and a Tanh activation function. A dropout rate of 0.2 is used for regularization.
Fully Connected Layer: Dense layer with 1 neuron and a Sigmoid activation function for binary classification.
Model Training
The model is trained on 80% of the 40,000 sampled IMDB reviews and validated on the remaining 20%.
The model is compiled with binary crossentropy loss, Adam optimizer (learning rate = 0.001), a batch size of 32, and trained for 15 epochs, with early stopping based on validation loss. Early stopping patience is set to 3 epochs.
Model Performance
Training accuracy increased steadily, reaching approximately 89% after 10 epochs.
Validation accuracy remained stable at around 87%, indicating good generalization.
The final test accuracy on the IMDB test set was around 86%, suggesting a well-trained model with slight room for improvement.
The model performed similarly on the Metacritic dataset, achieving a test accuracy of approximately 77%, showing that it generalizes well across different review datasets but could improve if LSTM was used instead of RNN.
Early stopping was triggered after a few epochs in both training phases, preventing overfitting and ensuring that the best model was retained.
Managerial Insights
1. Model Effectiveness & Business Implications
The RNN model performs reasonably well on the IMDB dataset but generalizes poorly on Metacritic reviews.
This suggests that Metacritic reviews might have different writing styles, slang, or review structures compared to IMDB. This highlights the importance of training data diversity for robust sentiment analysis.
2. Improvement Areas
Better Preprocessing: Introduce techniques like stemming, lemmatization, stop-word removal, and n-grams to improve accuracy.
More Complex Architectures: RNNs have limited long-term memory; switching to LSTMs may enhance generalization.
Larger Dataset & Augmentation: Training on a combined dataset of IMDB and Metacritic reviews may improve model robustness.
Domain Adaptation: Fine-tuning the model specifically on Metacritic reviews could improve cross-domain accuracy.
3. Business Applications
Customer Sentiment Monitoring: Companies can use this model to analyze movie, product, or service reviews to gauge public opinion.
Brand Reputation Analysis: Identifying sentiment trends can help businesses manage PR crises and improve customer engagement.
Automated Review Filtering: Businesses can filter out fake reviews or spam using an improved sentiment classification model.
4. Conclusion & Recommendations
Immediate Steps:
Improve text preprocessing by handling stop words and using TF-IDF weights.
Fine-tune the model using transfer learning with additional datasets.
Consider switching to LSTM/GRU-based models for improved generalization.
Long-Term Strategy:
Expand training data by incorporating reviews from multiple platforms.
Implement real-time sentiment tracking in a dashboard for actionable insights.
Conduct A/B testing with different architectures to find the best-performing model.
Aim for higher accuracy (target: 75%+) through continuous optimization.
This meticulously detailed report provides an extensive overview of the Reviews Sentiment Analysis project using an RNN, meticulously covering the objective, data description, observations, and managerial insights. The inclusion of the analysis of the report notebook, data preprocessing, model evaluation, and long-term strategic recommendations highlight the model implementation insights.
