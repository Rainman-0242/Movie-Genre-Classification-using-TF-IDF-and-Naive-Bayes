
# 🎬 Movie Genre Classification using TF-IDF and Machine Learning Models

## 📌 Overview

This project builds a machine learning model that predicts the **genre** of a movie based on its **plot summary** or **textual description**. It combines **Natural Language Processing (NLP)** with machine learning models using **TF-IDF** for feature extraction.

This model can be used in:
- 🎯 Recommendation engines
- 🎥 Content categorization
- 🔎 Search and filtering systems in movie databases

---

## 🧠 Objectives

- Clean and preprocess raw movie descriptions
- Convert text into numerical features using **TF-IDF**
- Train and compare multiple models: **Naive Bayes**, **SVC**, and **SVR**
- Evaluate model performance with accuracy and classification metrics
   (./download(11).png)
   (./download(12).png)
---

## 📂 Dataset

The dataset contains:
- **Title** of the movie
- **Genre** of the movie (target)
- **Description** of the movie (used for prediction)
(./)

The dataset is stored in a `.txt` file and loaded using custom delimiters (`:::`).

---

## ⚙️ Technologies Used

- Python 🐍
- Pandas
- Scikit-learn
- NLTK (for text preprocessing)
- Matplotlib & Seaborn (for visualization)

---

## 🚀 Model Pipeline

1. **Text Preprocessing**  
   - Lowercasing  
   - Removing punctuation and special characters  
   - Tokenization  
   - Stopword removal  
   - Stemming (Lancaster Stemmer)

2. **TF-IDF Vectorization**  
   Convert processed descriptions into a numerical format using `TfidfVectorizer`.

3. **Train/Test Split**  
   The dataset is split using `train_test_split`.

4. **Model Training**  
   The models trained include:
   - `MultinomialNB` (Naive Bayes)
   - `SVC` (Support Vector Classification)
   - `SVR` (Support Vector Regression)

5. **Evaluation**  
   Model performance is compared using accuracy, precision, recall, and F1-score.

---

## 📊 Results

- Each model provides different insights; SVC performs well for classification tasks, while SVR can capture ranking-based interpretations.
- Naive Bayes offers fast and interpretable results for baseline evaluation.
(./download(13).png)

---

## 📎 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/movie-genre-classification.git
   cd movie-genre-classification
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook movie_genre_classification.ipynb
   ```

---

## 📌 Future Work

- Handle multi-label classification (movies with multiple genres)
- Try advanced NLP techniques (e.g., BERT, LSTM)
- Improve text preprocessing with lemmatization

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
