from flask import Flask, render_template, request, redirect, url_for, session
import requests
import html
import random

app = Flask(__name__)
app.secret_key = "supersecretkey"

TOTAL_QUESTIONS = 10


def fetch_questions():
    url = f"https://opentdb.com/api.php?amount={TOTAL_QUESTIONS}&type=multiple"
    response = requests.get(url).json()

    questions = []

    for item in response["results"]:
        answers = item["incorrect_answers"] + [item["correct_answer"]]
        random.shuffle(answers)

        questions.append({
            "question": html.unescape(item["question"]),
            "correct": html.unescape(item["correct_answer"]),
            "answers": [html.unescape(a) for a in answers]
        })

    return questions


@app.route("/")
def index():
    session["questions"] = fetch_questions()
    session["current"] = 0
    session["score"] = 0
    session["results"] = []

    return redirect(url_for("question"))


@app.route("/question", methods=["GET", "POST"])
def question():
    questions = session.get("questions", [])
    current = session.get("current", 0)
    score = session.get("score", 0)
    results = session.get("results", [])

    if request.method == "POST":

        selected = request.form.get("answer")
        current_question = questions[current]

        is_correct = selected == current_question["correct"]

        if is_correct:
            score += 1
            session["score"] = score

        # Save detailed result
        results.append({
            "question": current_question["question"],
            "answer": selected,
            "correct_answer": current_question["correct"],
            "is_correct": is_correct
        })

        session["results"] = results

        current += 1
        session["current"] = current

        if current >= TOTAL_QUESTIONS:
            return redirect(url_for("result"))

    if current >= TOTAL_QUESTIONS:
        return redirect(url_for("result"))

    return render_template(
        "question.html",
        question=questions[current],
        current=current + 1,
        total=TOTAL_QUESTIONS
    )


@app.route("/result")
def result():
    score = session.get("score", 0)
    results = session.get("results", [])

    return render_template(
        "result.html",
        score=score,
        results=results
    )


if __name__ == "__main__":
    app.run(debug=True)