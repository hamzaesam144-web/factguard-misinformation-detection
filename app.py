import hashlib
import json
import math
import os
import re
import sqlite3
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import requests
import tldextract
import torch
from dotenv import load_dotenv
from flask import Flask, g, jsonify, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from dateutil import parser as date_parser

    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False

load_dotenv()

app = Flask(__name__)

DATABASE = "posts.db"
CLASSIFIER_PATH = "factguard_mmbert_small"
NLI_MODEL_NAME = "BalaRajesh1/mmbert-small-nli"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def get_db():
    db = getattr(g, "_database", None)

    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.cursor().execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                score REAL,
                misinfo INTEGER DEFAULT 0
            )
        """)
        db.commit()

    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)

    if db is not None:
        db.close()


@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers",
                         "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods",
                         "GET,PUT,POST,DELETE,OPTIONS")
    return response


print("Loading main classifier...")

classifier_tokenizer = AutoTokenizer.from_pretrained(
    CLASSIFIER_PATH, local_files_only=True
)

classifier_model = AutoModelForSequenceClassification.from_pretrained(
    CLASSIFIER_PATH, local_files_only=True
).to(device)

classifier_model.eval()

print("Main classifier loaded")

print("Loading NLI model...")

nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(
    device
)

nli_model.eval()

print("NLI labels:", nli_model.config.id2label)
print("NLI model loaded")

with open("trusted_websites.json", "r", encoding="utf-8") as f:
    TRUSTED_DOMAINS = set(json.load(f))


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


def normalize_text(text):
    text = str(text).lower()

    text = re.sub(r"[أإآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_domain(url):
    extracted = tldextract.extract(url)
    return f"{extracted.domain}.{extracted.suffix}"


def calculate_model_score(text):
    inputs = classifier_tokenizer(
        text, truncation=True, padding=True, max_length=256, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = classifier_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    fake_prob = probs[1]

    return fake_prob


def search(query):
    api_key = os.getenv("TVLY_API")

    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}

    data = {
        "query": query
        + ' true or false (English response, use the words "true" or "false" in the response)',
        "search_depth": "advanced",
        "max_results": 10,
        "include_answer": "advanced",
    }

    url = "https://api.tavily.com/search"

    response = requests.post(url, headers=headers, json=data, timeout=20)
    payload = response.json()
    return payload


def check_agreement(premise, hypothesis):
    inputs = nli_tokenizer(
        premise, hypothesis, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = nli_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    labels = nli_model.config.id2label

    output = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}

    for i, label in labels.items():
        label = label.lower()

        if "entail" in label:
            output["entailment"] = probs[i]
        elif "neutral" in label:
            output["neutral"] = probs[i]
        elif "contrad" in label:
            output["contradiction"] = probs[i]

    return output


def extract_temporal_reference(text, reference_time=None, absolute_only=False):
    if not text:
        return None

    if reference_time is None:
        reference_time = datetime.now()

    # --- Relative terms (skipped when absolute_only=True) ---
    if not absolute_only:
        relative_en = {
            r"\btoday\b": timedelta(days=0),
            r"\byesterday\b": timedelta(days=-1),
            r"\btomorrow\b": timedelta(days=1),
            r"\blast\s+week\b": timedelta(weeks=-1),
            r"\blast\s+month\b": timedelta(days=-30),
            r"\bthis\s+week\b": timedelta(days=0),
            r"\bnext\s+week\b": timedelta(weeks=1),
            r"\bnext\s+month\b": timedelta(days=30),
        }

        relative_ar = {
            r"\bاليوم\b": timedelta(days=0),
            r"\bأمس\b": timedelta(days=-1),
            r"\bامس\b": timedelta(days=-1),
            r"\bغداً\b": timedelta(days=1),
            r"\bغدا\b": timedelta(days=1),
            r"\bالأسبوع\s*الماضي\b": timedelta(weeks=-1),
            r"\bالاسبوع\s*الماضي\b": timedelta(weeks=-1),
            r"\bالشهر\s*الماضي\b": timedelta(days=-30),
            r"\bالأسبوع\s*القادم\b": timedelta(weeks=1),
            r"\bالاسبوع\s*القادم\b": timedelta(weeks=1),
            r"\bالشهر\s*القادم\b": timedelta(days=30),
        }

        for pattern, delta in {**relative_en, **relative_ar}.items():
            if re.search(pattern, text, re.IGNORECASE):
                return reference_time + delta

    # --- Arabic month names ---
    arabic_months_standard = {
        "يناير": 1,
        "فبراير": 2,
        "مارس": 3,
        "أبريل": 4,
        "ابريل": 4,
        "مايو": 5,
        "يونيو": 6,
        "يوليو": 7,
        "أغسطس": 8,
        "اغسطس": 8,
        "سبتمبر": 9,
        "أكتوبر": 10,
        "اكتوبر": 10,
        "نوفمبر": 11,
        "ديسمبر": 12,
    }
    arabic_months_levantine = {
        "كانون الثاني": 1,
        "شباط": 2,
        "آذار": 3,
        "اذار": 3,
        "نيسان": 4,
        "أيار": 5,
        "ايار": 5,
        "حزيران": 6,
        "تموز": 7,
        "آب": 8,
        "اب": 8,
        "أيلول": 9,
        "ايلول": 9,
        "تشرين الأول": 10,
        "تشرين الاول": 10,
        "تشرين الثاني": 11,
        "كانون الأول": 12,
        "كانون الاول": 12,
    }
    all_arabic_months = {**arabic_months_standard, **arabic_months_levantine}

    # Try two-word month names first (e.g. "تشرين الأول"), then single-word
    for month_name in sorted(all_arabic_months, key=len, reverse=True):
        month_num = all_arabic_months[month_name]
        # Pattern: optional day + month name + optional year
        pattern = r"(?:(\d{1,2})\s+)?" + \
            re.escape(month_name) + r"(?:\s+(\d{4}))?"
        m = re.search(pattern, text)
        if m:
            day = int(m.group(1)) if m.group(1) else 1
            year = int(m.group(2)) if m.group(2) else reference_time.year
            try:
                return datetime(year, month_num, min(day, 28))
            except ValueError:
                return datetime(year, month_num, 1)

    # --- English month names: "January 5", "5 June", "March 2025" ---
    en_months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "sept": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    en_month_pattern = "|".join(en_months.keys())

    # "June 3", "June 3, 2025", "June 2025"
    m = re.search(
        r"\b(" + en_month_pattern + r")\s+(\d{1,2})(?:\s*,?\s*(\d{4}))?\b",
        text,
        re.IGNORECASE,
    )
    if m:
        month_num = en_months[m.group(1).lower()]
        day = int(m.group(2))
        if day > 31:
            # Probably "March 2025" parsed wrong; treat group(2) as year
            year = day
            day = 1
        else:
            year = int(m.group(3)) if m.group(3) else reference_time.year
        try:
            return datetime(year, month_num, min(day, 28))
        except ValueError:
            return datetime(year, month_num, 1)

    # "3 June", "3 June 2025"
    m = re.search(
        r"\b(\d{1,2})\s+(" + en_month_pattern + r")(?:\s+(\d{4}))?\b",
        text,
        re.IGNORECASE,
    )
    if m:
        day = int(m.group(1))
        month_num = en_months[m.group(2).lower()]
        year = int(m.group(3)) if m.group(3) else reference_time.year
        try:
            return datetime(year, month_num, min(day, 28))
        except ValueError:
            return datetime(year, month_num, 1)

    # --- Numeric date formats ---
    # yyyy-mm-dd or yyyy/mm/dd
    m = re.search(r"\b(\d{4})[\-/](\d{1,2})[\-/](\d{1,2})\b", text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # dd/mm/yyyy or dd-mm-yyyy (day/month — Arabic-region convention)
    m = re.search(r"\b(\d{1,2})[\-/](\d{1,2})[\-/](\d{4})\b", text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(year, month, day)
        except ValueError:
            pass

    # Short d/m or d-m (assume current year, day/month order)
    m = re.search(r"\b(\d{1,2})[\-/](\d{1,2})\b", text)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12 and 1 <= day <= 31:
            try:
                return datetime(reference_time.year, month, day)
            except ValueError:
                pass

    # --- Fallback: dateutil parser (if available) ---
    if HAS_DATEUTIL:
        try:
            parsed = date_parser.parse(
                text, fuzzy=True, default=reference_time)
            # Only accept if it actually found something different from the default
            if parsed != reference_time:
                return parsed
        except (ValueError, OverflowError):
            pass

    return None


def compute_temporal_relevance(claim_date, evidence_date):
    if claim_date is None or evidence_date is None:
        return 1.0

    diff_days = abs((claim_date - evidence_date).days)

    if diff_days == 0:
        return 1.2
    elif diff_days <= 3:
        return 1.1
    elif diff_days <= 7:
        return 1.0
    elif diff_days <= 30:
        return 0.85
    elif diff_days <= 90:
        return 0.7
    else:
        return 0.5


def calculate_final_score(model_prediction, results, original_text):
    claim_date = extract_temporal_reference(
        original_text, reference_time=datetime.now()
    )

    h = model_prediction
    for i, result in enumerate(results):
        title = result["title"]
        content = result["content"]
        url = result["url"]

        base_domain = get_domain(url)
        if base_domain in TRUSTED_DOMAINS:
            trust_factor = 0.3
        else:
            trust_factor = 0.1

        # Temporal relevance: dampen results about a different time period
        evidence_date = extract_temporal_reference(
            title + content, absolute_only=True)
        temporal_factor = compute_temporal_relevance(claim_date, evidence_date)

        relevance = temporal_factor * trust_factor

        eps = 0.05

        probs = check_agreement(title, original_text)
        entailment = probs["entailment"]
        contradiction = probs["contradiction"]
        p_e_given_fake = (contradiction * relevance) + \
            (0.5 * (1.0 - relevance)) + eps
        p_e_given_real = (entailment * relevance) + \
            (0.5 * (1.0 - relevance)) + eps
        numerator = p_e_given_fake * h
        denominator = numerator + (p_e_given_real * (1.0 - h))
        h = numerator / denominator

        probs = check_agreement(content, original_text)
        entailment = probs["entailment"]
        contradiction = probs["contradiction"]
        p_e_given_fake = (contradiction * relevance) + \
            (0.5 * (1.0 - relevance)) + eps
        p_e_given_real = (entailment * relevance) + \
            (0.5 * (1.0 - relevance)) + eps
        numerator = p_e_given_fake * h
        denominator = numerator + (p_e_given_real * (1.0 - h))
        h = numerator / denominator

        print(f"new h = {h}")

    final_score = h

    print("  Final P(fake):", final_score)

    return final_score


@app.route("/predict_search", methods=["POST", "OPTIONS"])
def predict_search():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    text = request.json["text"].strip()

    model_prediction = calculate_model_score(text)

    results = search(text)

    if "false" in results["answer"].lower():
        model_prediction = 0.9
    elif "true" in results["answer"].lower():
        model_prediction = 0.1

    final_score = calculate_final_score(
        model_prediction, results["results"], text
    )

    return jsonify(
        {
            "score": final_score,
            "results": results["results"],
        }
    )


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    text = request.json.get("text", "").strip()

    if not text:
        return jsonify({"score": 0.0, "error": "No text provided"}), 400

    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

    cursor = get_db().cursor()

    cursor.execute("SELECT score FROM posts WHERE id = ?", (text_hash,))

    row = cursor.fetchone()

    if row:
        return jsonify({"score": row[0]})

    prediction = calculate_model_score(text)

    cursor.execute(
        "INSERT INTO posts (id, content, score) VALUES (?, ?, ?)",
        (text_hash, text, prediction),
    )

    get_db().commit()

    return jsonify({"score": prediction})


@app.route("/report", methods=["POST", "OPTIONS"])
def report():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    text = request.json.get("text", "").strip()

    if not text:
        return jsonify({"success": False, "error": "No text provided"}), 400

    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT id FROM posts WHERE id = ?", (text_hash,))

    if cursor.fetchone():
        cursor.execute(
            "UPDATE posts SET misinfo = 1 WHERE id = ?", (text_hash,))

    else:
        prediction = calculate_model_score(text)

        cursor.execute(
            "INSERT INTO posts (id, content, score, misinfo) VALUES (?, ?, ?, 1)",
            (text_hash, text, prediction),
        )

    db.commit()

    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
