from functools import lru_cache

from flask import Flask, render_template, request, redirect, url_for, session
import requests
import html
import random
import translate


app = Flask(__name__)
app.secret_key = "supersecretkey"

TOTAL_QUESTIONS = 10


@lru_cache(maxsize=1000)
def translate_text(text: str) -> str:
    """Translate and cache a quiz string for the lifetime of this Flask process."""
    print(f"Translating: {text}")
    return translate.translate_to_latvian(text)


def translate_question(question_text: str, correct_answer: str, incorrect_answers: list[str]):
    """Translate a question and all of its answers before shuffling the answers."""
    translated_question = translate_text(question_text)
    translated_correct = translate_text(correct_answer)
    translated_incorrect = [translate_text(answer) for answer in incorrect_answers]

    answers = translated_incorrect + [translated_correct]
    random.shuffle(answers)

    return {
        "question": translated_question,
        "correct": translated_correct,
        "answers": answers,
    }


def fetch_questions():
    url = f"https://opentdb.com/api.php?amount={TOTAL_QUESTIONS}&type=multiple"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    questions = []

    for item in data["results"]:
        # Open Trivia DB returns HTML entities such as &quot; and &#039;.
        # Decode them before sending the strings to the translation model.
        question_text = html.unescape(item["question"])
        correct_answer = html.unescape(item["correct_answer"])
        incorrect_answers = [
            html.unescape(answer)
            for answer in item["incorrect_answers"]
        ]

        translated = translate_question(
            question_text,
            correct_answer,
            incorrect_answers,
        )

        questions.append(translated)

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

        results.append({
            "question": current_question["question"],
            "answer": selected,
            "correct_answer": current_question["correct"],
            "is_correct": is_correct,
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
        total=TOTAL_QUESTIONS,
    )


@app.route("/result")
def result():
    score = session.get("score", 0)
    results = session.get("results", [])

    return render_template(
        "result.html",
        score=score,
        results=results,
    )


if __name__ == "__main__":
    # Important for a 30B model: Flask's development reloader launches another
    # Python process and can cause the model to be loaded twice. Keep debugging
    # enabled if desired, but disable the reloader.
    app.run(debug=True, use_reloader=False)
