from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import fitz  # PyMuPDF

def pdf_to_text(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Initialize a text string
    text = ""
    
    # Loop through each page
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)  # Load each page
        text += page.get_text()  # Extract text from the page
    
    return text

# Path to the PDF file
pdf_path = '"C:\Users\91821\Downloads\Carlsson_HEAL-SWIN_A_Vision_Transformer_On_The_Sphere_CVPR_2024_paper.pdf"'

# Convert PDF to text
documents = pdf_to_text(pdf_path)



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
