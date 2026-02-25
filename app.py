from flask import Flask, render_template, request
import PyPDF2
import base64
from resume_analyzer import analyze_resume

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    skills = []
    details = None
    photo_data_url = None
    skill_matches = []
    job_match = None

    if request.method == "POST":
        file = request.files["resume"]
        photo = request.files.get("photo")
        job_description = request.form.get("job_description") or None

        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        score, skills, details, skill_matches, job_match = analyze_resume(text, job_description)

        if photo and photo.filename:
            mime = (photo.mimetype or "").lower()
            if mime.startswith("image/"):
                data = photo.read()
                if data:
                    b64 = base64.b64encode(data).decode("ascii")
                    photo_data_url = f"data:{mime};base64,{b64}"

    return render_template(
        "index.html",
        score=score,
        skills=skills,
        details=details,
        skill_matches=skill_matches,
        job_match=job_match,
        photo_data_url=photo_data_url,
    )

if __name__ == "__main__":
    app.run(debug=True)