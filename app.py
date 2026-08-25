from flask import Flask, render_template, request
from utils import process_pdf, answer_question

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    status = None

    if request.method == "POST":
        if 'pdf_file' in request.files:
            pdf = request.files['pdf_file']
            pdf.save("uploaded.pdf")
            process_pdf("uploaded.pdf")
            status = "✅ PDF uploaded and processed."

        elif 'query' in request.form:
            query = request.form['query']
            answer = answer_question(query)

    return render_template("index.html", answer=answer, status=status)

if __name__ == "__main__":
    app.run(debug=True)
