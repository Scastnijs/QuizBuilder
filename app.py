from datetime import datetime
import html
import random
import re

from flask import Flask, render_template, request, redirect, url_for, session
import requests
import translate


app = Flask(__name__)
app.secret_key = "supersecretkey"

TOTAL_QUESTIONS = 10
API_QUESTIONS = TOTAL_QUESTIONS + 3


# English month names accepted in trivia answers.
MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _valid_full_date(day: int, month: int, year: int):
    """Return DD/MM/YYYY when valid, otherwise None."""
    try:
        value = datetime(year, month, day)
        return value.strftime("%d/%m/%Y")
    except ValueError:
        return None


def normalize_date_answer(answer: str):
    """
    Normalize recognized English/numeric dates to:
      DD/MM/YYYY
      DD/MM
      MM/YYYY

    Returns None when the answer is not recognized as a date.
    """
    value = answer.strip()

    # YYYY-MM-DD
    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", value)
    if match:
        year, month, day = map(int, match.groups())
        return _valid_full_date(day, month, year)

    # Month name DD, YYYY   e.g. July 4, 1776
    match = re.fullmatch(
        r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        month_name, day, year = match.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            return _valid_full_date(int(day), month, int(year))

    # DD Month name YYYY   e.g. 4 July 1776
    match = re.fullmatch(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)[,]?\s+(\d{4})",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        day, month_name, year = match.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            return _valid_full_date(int(day), month, int(year))

    # Month name DD   e.g. July 4
    match = re.fullmatch(
        r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        month_name, day = match.groups()
        month = MONTHS.get(month_name.lower())
        day = int(day)
        if month:
            try:
                datetime(2000, month, day)
                return f"{day:02d}/{month:02d}"
            except ValueError:
                return None

    # DD Month name   e.g. 4 July
    match = re.fullmatch(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        day, month_name = match.groups()
        month = MONTHS.get(month_name.lower())
        day = int(day)
        if month:
            try:
                datetime(2000, month, day)
                return f"{day:02d}/{month:02d}"
            except ValueError:
                return None

    # Month name YYYY   e.g. July 1776
    match = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", value, flags=re.IGNORECASE)
    if match:
        month_name, year = match.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            return f"{month:02d}/{int(year):04d}"

    # MM/YYYY
    match = re.fullmatch(r"(\d{1,2})[/-](\d{4})", value)
    if match:
        month, year = map(int, match.groups())
        if 1 <= month <= 12:
            return f"{month:02d}/{year:04d}"

    # Numeric full date with / or -
    # If first part > 12, assume DD/MM/YYYY.
    # Otherwise assume English-source MM/DD/YYYY.
    match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", value)
    if match:
        first, second, year = map(int, match.groups())

        if first > 12:
            day, month = first, second
        else:
            month, day = first, second

        return _valid_full_date(day, month, year)

    return None


def is_numeric_time_or_money_answer(answer: str) -> bool:
    """
    Return True for answers that should be preserved exactly:
    numbers, percentages, fractions, times, and money values.
    """
    value = answer.strip()

    # Integer/decimal values, optional sign and thousands separators.
    number_pattern = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

    if re.fullmatch(number_pattern, value):
        return True

    # Percentages, e.g. 25%, -3.5%
    if re.fullmatch(number_pattern + r"\s*%", value):
        return True

    # Fractions, e.g. 1/2, 10/25
    if re.fullmatch(r"[+-]?\d+\s*/\s*\d+", value):
        return True

    # Times: 08:30, 8:30 PM, 23:59:30, 8 PM, 8AM
    if re.fullmatch(
        r"(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s*[APap][Mm])?",
        value,
    ):
        return True

    if re.fullmatch(r"(?:0?[1-9]|1[0-2])\s*[APap][Mm]", value):
        return True

    # Currency symbols before or after amount.
    if re.fullmatch(rf"[$€£¥₹]\s*{number_pattern}", value):
        return True

    if re.fullmatch(rf"{number_pattern}\s*[$€£¥₹]", value):
        return True

    # Common currency codes before or after amount.
    currency_codes = r"(?:USD|EUR|GBP|JPY|CNY|CAD|AUD|CHF|INR)"
    if re.fullmatch(rf"{currency_codes}\s+{number_pattern}", value, re.IGNORECASE):
        return True

    if re.fullmatch(rf"{number_pattern}\s+{currency_codes}", value, re.IGNORECASE):
        return True

    return False


def prepare_answer_for_batch(answer: str) -> dict:
    """
    Normalize one answer before the single quiz-item LLM call.

    Returns a dict with:
      text      - normalized/original answer text
      translate - whether TildeOpen should translate this answer
    """
    normalized_date = normalize_date_answer(answer)
    if normalized_date is not None:
        print(f"Date answer: {answer} -> {normalized_date}")
        return {"text": normalized_date, "translate": False}

    if is_numeric_time_or_money_answer(answer):
        print(f"Keeping numeric/time/money answer: {answer}")
        return {"text": answer, "translate": False}

    return {"text": answer, "translate": True}


def translate_question(question_text: str, correct_answer: str, incorrect_answers: list[str]):
    """
    Normalize all answers first, then translate the complete quiz item in ONE
    TildeOpen generation. With 10 questions this reduces normal translation work
    from up to ~50 generations to exactly 10 generations.
    """
    source_answers = incorrect_answers + [correct_answer]
    prepared_answers = []
    correct_id = None

    for index, answer in enumerate(source_answers):
        answer_id = f"A{index}"
        prepared = prepare_answer_for_batch(answer)

        prepared_answers.append({
            "id": answer_id,
            "text": prepared["text"],
            "translate": prepared["translate"],
        })

        if index == len(source_answers) - 1:
            correct_id = answer_id

    print(f"Translating complete quiz item in one generation: {question_text}")
    translated = translate.translate_quiz_item(
        question=question_text,
        answers=prepared_answers,
    )

    translated_answers_by_id = translated["answers"]
    prepared_correct = translated_answers_by_id[correct_id]
    answers = [
        translated_answers_by_id[item["id"]]
        for item in prepared_answers
    ]
    random.shuffle(answers)

    return {
        "question": translated["question"],
        "correct": prepared_correct,
        "answers": answers,
    }

def fetch_questions():
    """
    Request 13 candidate questions from Open Trivia DB and keep translating
    candidates until 10 successful Latvian quiz questions have been collected.

    A TildeOpen parsing ValueError skips only the failed candidate. The next API
    question is then tried. This gives up to 3 spare candidates while still
    returning exactly TOTAL_QUESTIONS questions.
    """
    url = f"https://opentdb.com/api.php?amount={API_QUESTIONS}&type=multiple"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    questions = []
    skipped = 0

    for candidate_number, item in enumerate(data["results"], start=1):
        # Stop immediately once the required 10 successful questions exist.
        if len(questions) >= TOTAL_QUESTIONS:
            break

        question_text = html.unescape(item["question"])
        correct_answer = html.unescape(item["correct_answer"])
        incorrect_answers = [
            html.unescape(answer)
            for answer in item["incorrect_answers"]
        ]

        try:
            translated_question = translate_question(
                question_text,
                correct_answer,
                incorrect_answers,
            )
        except ValueError as exc:
            skipped += 1
            print(
                f"Skipping candidate question {candidate_number}/{API_QUESTIONS} "
                f"because translation parsing failed."
            )
            print(f"Question: {question_text}")
            print(f"Error: {exc}")
            print(
                f"Successful questions: {len(questions)}/{TOTAL_QUESTIONS}; "
                f"skipped: {skipped}/3"
            )
            continue

        questions.append(translated_question)

        print(
            f"Accepted candidate question {candidate_number}/{API_QUESTIONS}. "
            f"Successful questions: {len(questions)}/{TOTAL_QUESTIONS}"
        )

    if len(questions) < TOTAL_QUESTIONS:
        raise RuntimeError(
            f"Could build only {len(questions)} of {TOTAL_QUESTIONS} required "
            f"quiz questions from {API_QUESTIONS} API candidates. "
            f"{skipped} candidate translation(s) failed."
        )

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
    app.run(debug=True, use_reloader=False)
