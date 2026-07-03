from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# CSV file read pannudhu
data = pd.read_csv("internship.csv")

# Skills column edukkudhu
skills = data["Skills"]

# TF-IDF vector create pannudhu
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(skills)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    name = request.form["name"]
    degree = request.form["degree"]
    skill = request.form["skill"]

    student_skill = [skill]

    student_vector = vectorizer.transform(student_skill)

    similarity = cosine_similarity(student_vector, vectors)

    index = similarity.argmax()

    company = data.iloc[index]["Company"]

    return f"""
    <h2>Internship Recommendation Result</h2>
    <p><b>Student Name:</b> {name}</p>
    <p><b>Degree:</b> {degree}</p>
    <p><b>Skills:</b> {skill}</p>
    <p><b>Recommended Company:</b> {company}</p>
    <br>
    <a href="/">Go Back</a>
    """

if __name__ == "__main__":
    app.run(debug=True)