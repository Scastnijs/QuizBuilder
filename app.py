from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
import threading
import uuid
import html
import random
import re

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import requests
import translate


app = Flask(__name__)
app.secret_key = "supersecretkey"

TOTAL_QUESTIONS = 10
API_QUESTIONS = TOTAL_QUESTIONS + 10
SUPPORTED_LANGUAGES = {"en", "lv", "de", "ua", "bg", "cz", "ee", "fi", "fr", "hu", "is", "it", "lt", "nl", "pl", "pt", "ro", "ru", "se", "si", "sk", "tr", "rs", "es"}
DEFAULT_LANGUAGE = "en"
TEXTS_FILE = Path(__file__).with_name("texts.json")

with TEXTS_FILE.open("r", encoding="utf-8") as file:
    TEXTS = json.load(file)


def get_text(language=None):
    """Return the UI text group for a supported language."""
    selected = language or session.get("language", DEFAULT_LANGUAGE)
    if selected not in SUPPORTED_LANGUAGES:
        selected = DEFAULT_LANGUAGE
    return TEXTS[selected]

# In-memory quiz build jobs. Suitable for this local single-process QuizBuilder app.
_build_jobs = {}
_build_jobs_lock = threading.Lock()


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


def prepare_answer(answer: str) -> str:
    """Normalize deterministic answer types; leave all other answers untranslated."""
    normalized_date = normalize_date_answer(answer)
    if normalized_date is not None:
        print(f"Date answer: {answer} -> {normalized_date}")
        return normalized_date

    if is_numeric_time_or_money_answer(answer):
        print(f"Keeping numeric/time/money answer: {answer}")
        return answer

    return answer


def translate_question(question_text: str, correct_answer: str, incorrect_answers: list[str], language: str):
    """Translate only the question. Answers are normalized locally and never sent to TildeOpen."""
    started = perf_counter()
    print(f"Translating question: {question_text}")

    translated_question = translate.translate_text(question_text, language)

    prepared_correct = prepare_answer(correct_answer)
    answers = [prepare_answer(answer) for answer in incorrect_answers]
    answers.append(prepared_correct)
    random.shuffle(answers)

    elapsed = perf_counter() - started
    print(f"Translated question: {translated_question}")
    print(f"Question translation time: {elapsed:.3f} s")

    # Keep the complete TildeOpen output in the terminal above, but only show
    # text through the first question mark in the quiz UI. If TildeOpen did
    # not produce a question mark, keep the complete translation unchanged.
    # If TildeOpen produced a single period, keep only the text through that period.
    first_question_mark = translated_question.find("?")

    if first_question_mark != -1:
        # Preferred case: keep everything through the first question mark.
        display_question = translated_question[:first_question_mark + 1].strip()
    elif translated_question.count(".") == 1:
        # Fallback: if there is no question mark and exactly one period,
        # keep everything through that period.
        first_period = translated_question.find(".")
        display_question = translated_question[:first_period + 1].strip()
    else:
        # Always initialize display_question, even when neither rule matches.
        display_question = translated_question.strip()

    # Validate length only after extracting the question shown in the quiz.
    if len(display_question) > 300:
        raise ValueError(
            f"Translated question is too long after cutting: "
            f"{len(display_question)} characters (maximum 300)."
        )

    return {
        "question": display_question,
        "correct": prepared_correct,
        "answers": answers,
    }

def build_question(question_text: str, correct_answer: str, incorrect_answers: list[str], language: str):
    """Prepare one quiz question. TildeOpen is used for supported non-English languages."""
    if language != "en":
        return translate_question(
            question_text, correct_answer, incorrect_answers, language
        )

    # English: use the API question directly. TildeOpen is never loaded/called.
    prepared_correct = prepare_answer(correct_answer)
    answers = [prepare_answer(answer) for answer in incorrect_answers]
    answers.append(prepared_correct)
    random.shuffle(answers)

    print(f"English question (no translation): {question_text}")
    return {
        "question": question_text,
        "correct": prepared_correct,
        "answers": answers,
    }


def fetch_questions(language: str, progress_callback=None):
    """Build 10 questions in the selected language."""
    url = f"https://opentdb.com/api.php?amount={API_QUESTIONS}&type=multiple"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    questions = []
    skipped = 0

    for candidate_number, item in enumerate(data["results"], start=1):
        if len(questions) >= TOTAL_QUESTIONS:
            break

        question_text = html.unescape(item["question"])
        correct_answer = html.unescape(item["correct_answer"])
        incorrect_answers = [html.unescape(answer) for answer in item["incorrect_answers"]]

        try:
            prepared_question = build_question(
                question_text, correct_answer, incorrect_answers, language
            )
        except ValueError as exc:
            # Translation parsing failures are relevant to translated-language modes.
            skipped += 1
            print(
                f"Skipping candidate question {candidate_number}/{API_QUESTIONS} "
                f"because translation parsing failed."
            )
            print(f"Question: {question_text}")
            print(f"Error: {exc}")
            print(
                f"Successful questions: {len(questions)}/{TOTAL_QUESTIONS}; "
                f"skipped: {skipped}/{API_QUESTIONS - TOTAL_QUESTIONS}"
            )
            continue

        questions.append(prepared_question)

        if progress_callback is not None:
            progress_callback(len(questions), TOTAL_QUESTIONS)

        print(
            f"Accepted candidate question {candidate_number}/{API_QUESTIONS}. "
            f"Successful questions: {len(questions)}/{TOTAL_QUESTIONS}"
        )
        print(f"Question shown in quiz: {prepared_question['question']}")
        print("-" * 80)

    if len(questions) < TOTAL_QUESTIONS:
        raise RuntimeError(
            f"Could build only {len(questions)} of {TOTAL_QUESTIONS} required "
            f"quiz questions from {API_QUESTIONS} API candidates. "
            f"{skipped} candidate translation(s) failed."
        )

    return questions


def _run_quiz_build(job_id: str, language: str):
    """Build a quiz in a background thread and publish 0..10 progress."""
    def update_progress(completed: int, total: int):
        with _build_jobs_lock:
            job = _build_jobs.get(job_id)
            if job is not None:
                job["completed"] = completed
                job["total"] = total

    try:
        questions = fetch_questions(language, progress_callback=update_progress)
        with _build_jobs_lock:
            job = _build_jobs.get(job_id)
            if job is not None:
                job["questions"] = questions
                job["completed"] = TOTAL_QUESTIONS
                job["done"] = True
    except Exception as exc:
        print(f"Quiz build failed: {exc}")
        with _build_jobs_lock:
            job = _build_jobs.get(job_id)
            if job is not None:
                job["error"] = str(exc)
                job["done"] = True


@app.route("/")
def index():
    """Show language selection before starting a new quiz."""
    return render_template("language.html", text=get_text(DEFAULT_LANGUAGE))


@app.route("/start", methods=["GET", "POST"])
def start_quiz():
    """
    Start a new quiz.

    POST is used after language selection.
    GET is used by Play Again and keeps the language already stored in session.

    English skips loading.html because no TildeOpen translation is required.
    All non-English languages keep the background build and progress screen.
    """
    if request.method == "POST":
        language = request.form.get("language", DEFAULT_LANGUAGE)
    else:
        language = session.get("language", DEFAULT_LANGUAGE)

    if language not in SUPPORTED_LANGUAGES:
        language = DEFAULT_LANGUAGE

    session["language"] = language

    # English needs no TildeOpen translation, so build the quiz immediately
    # and go directly to question 1 without showing loading.html.
    if language == "en":
        questions = fetch_questions(language)
        session["questions"] = questions
        session["current"] = 0
        session["score"] = 0
        session["results"] = []
        return redirect(url_for("question"))

    # All non-English languages use TildeOpen, so build in the background and show progress.
    job_id = uuid.uuid4().hex

    with _build_jobs_lock:
        _build_jobs[job_id] = {
            "completed": 0,
            "total": TOTAL_QUESTIONS,
            "done": False,
            "error": None,
            "questions": None,
            "language": language,
        }

    thread = threading.Thread(
        target=_run_quiz_build,
        args=(job_id, language),
        daemon=True,
    )
    thread.start()

    return render_template(
        "loading.html",
        job_id=job_id,
        total=TOTAL_QUESTIONS,
        language=language,
        text=get_text(language),
    )


@app.route("/quiz-progress/<job_id>")
def quiz_progress(job_id):
    """Return the current translation progress for the loading page."""
    with _build_jobs_lock:
        job = _build_jobs.get(job_id)
        if job is None:
            return jsonify({"error": get_text()["job_not_found"]}), 404

        return jsonify({
            "completed": job["completed"],
            "total": job["total"],
            "done": job["done"],
            "error": job["error"],
        })


@app.route("/quiz-ready/<job_id>")
def quiz_ready(job_id):
    """Move the completed quiz into the browser session and start question 1."""
    with _build_jobs_lock:
        job = _build_jobs.get(job_id)
        if job is None:
            return get_text()["job_not_found"], 404
        if not job["done"]:
            return redirect(url_for("index"))
        if job["error"]:
            return f"{get_text()['generation_failed']}: {job['error']}", 500

        questions = job["questions"]
        del _build_jobs[job_id]

    session["questions"] = questions
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
        text=get_text(),
    )


@app.route("/result")
def result():
    score = session.get("score", 0)
    results = session.get("results", [])

    return render_template(
        "result.html",
        score=score,
        results=results,
        text=get_text(),
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
