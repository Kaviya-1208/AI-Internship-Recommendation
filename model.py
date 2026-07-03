import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# CSV file read pannudhu
data = pd.read_csv("internship.csv")

# Company skills edukkudhu
skills = data["Skills"]

# TF-IDF create pannudhu
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(skills)

# Student skill
student_skill = [input("Enter your skills:")]

# Student skill vector
student_vector = vectorizer.transform(student_skill)

# Similarity calculate pannudhu
similarity = cosine_similarity(student_vector, vectors)

# Highest matching company
index = similarity.argmax()

print("Recommended Company:")
print(data.iloc[index]["Company"])