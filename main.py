from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
documents = [
    "This is a sample document to extract keywords.",
    "TF-IDF is a statistical measure used to evaluate the importance of a word in a document.",
    "We will use TF-IDF to extract keywords from these documents."
]

# Initialize the TFIDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Convert the TFIDF matrix to a pandas DataFrame
df = pd.DataFrame(tfidf_matrix.T.toarray(), index=feature_names, columns=[f'Doc {i+1}' for i in range(len(documents))])

# Display the DataFrame
print(df)

# Extract keywords for each document
def extract_keywords(tfidf_vector, feature_names, top_n=5):
    sorted_indices = tfidf_vector.argsort()[::-1][:top_n]
    return [(feature_names[i], tfidf_vector[i]) for i in sorted_indices]

# Print the keywords for each document
for i in range(len(documents)):
    print(f"\nTop keywords in Doc {i+1}:")
    keywords = extract_keywords(tfidf_matrix[i].toarray()[0], feature_names)
    for keyword, score in keywords:
        print(f"{keyword}: {score:.4f}")
