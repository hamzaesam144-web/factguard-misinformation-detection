import os
import re
import glob
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


MODEL_NAME = "xlm-roberta-base"
OUTPUT_DIR = "factguard_xlm_roberta"
PREPARED_DIR = "prepared_data"

MAX_LEN = 256

# Practical setting: keeps training time reasonable.
MAX_PER_LABEL_PER_LANGUAGE = 3000
NUM_EPOCHS = 2
LEARNING_RATE = 1e-5

# If your hard_cases.csv contains label 2 = NEEDS_REVIEW,
# this binary training file will skip label 2 because the app/model is still binary:
# 0 = REAL, 1 = FAKE.
SKIP_NEEDS_REVIEW_IN_BINARY_TRAINING = True

# Stronger focus on the hard fake examples.
# These are added to TRAIN only after splitting, so validation/test stay fair.
HARD_FAKE_REPEAT_IN_TRAIN = 12
ADD_SYNTHETIC_HARD_FAKE = True
SYNTHETIC_HARD_FAKE_REPEAT = 4

os.makedirs(PREPARED_DIR, exist_ok=True)

TEXT_COLUMNS = [
    "Content", "content",
    "Text", "text",
    "title", "Title",
    "article", "Article",
    "body", "Body",
    "headline", "Headline"
]


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"with a Photo:.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text):
    text = str(text)
    arabic_chars = re.findall(r"[\u0600-\u06FF]", text)
    english_chars = re.findall(r"[a-zA-Z]", text)

    if len(arabic_chars) > len(english_chars):
        return "ar"

    return "en"


def infer_label_from_filename(file_path):
    name = os.path.basename(file_path).lower()

    # Important: check TRUE patterns before generic "misinfo",
    # because files like DataSet_Misinfo_TRUE contain the word "misinfo".
    true_keywords = [
        "misinfo_true",
        "true",
        "bbc",
        "aljazeera",
        "roya",
        "reuters"
    ]

    fake_keywords = [
        "misinfo_fake",
        "fake",
        "false",
        "misinformation"
    ]

    for word in true_keywords:
        if word in name:
            return 0

    for word in fake_keywords:
        if word in name:
            return 1

    return None


def read_csv_file(file_path):
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1256"]

    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception:
            pass

    return None


def find_text_columns(df):
    found = []

    for col in TEXT_COLUMNS:
        if col in df.columns:
            found.append(col)

    return found


def should_skip_file(file_path):
    path = file_path.replace("\\", "/").lower()

    skipped_parts = [
        "__pycache__",
        "prepared_data",
        "prepared_data_3class",
        "factguard_xlm_roberta",
        "factguard_xlm_roberta_3class",
        ".git",
        "static",
        "templates"
    ]

    return any(part in path for part in skipped_parts)


def parse_label_from_column(row):
    """
    Supports hard_cases.csv formats:
    - label = 0 / 1 / 2
    - label_name = REAL / FAKE / NEEDS_REVIEW
    - label = REAL / FAKE / TRUE / FALSE / NEEDS_REVIEW
    Returns:
    - 0 for REAL
    - 1 for FAKE
    - 2 for NEEDS_REVIEW
    - None if unknown
    """
    label_value = row.get("label", None)

    if pd.notna(label_value):
        value = str(label_value).strip().lower()

        if value in ["0", "real", "true", "likely_real"]:
            return 0

        if value in ["1", "fake", "false", "likely_fake"]:
            return 1

        if value in ["2", "needs_review", "review", "uncertain", "unknown", "unsupported"]:
            return 2

    label_name = row.get("label_name", None)

    if pd.notna(label_name):
        value = str(label_name).strip().lower()

        if value in ["real", "true", "likely_real"]:
            return 0

        if value in ["fake", "false", "likely_fake"]:
            return 1

        if value in ["needs_review", "review", "uncertain", "unknown", "unsupported"]:
            return 2

    return None


def load_all_files():
    rows = []

    # Read from ONE dataset folder only to avoid loading the same files twice
    # through both Dataset/ and dataset/. Keep your data inside Dataset/.
    search_patterns = [
        "Dataset/**/*.csv",
        "Dataset/**/*"
    ]

    files = []

    for pattern in search_patterns:
        files.extend(glob.glob(pattern, recursive=True))

    files = sorted(list(set(files)))

    print()
    print("Found files:", len(files))

    labeled_files_count = 0
    skipped_review_count = 0

    for file_path in files:
        if not os.path.isfile(file_path):
            continue

        if should_skip_file(file_path):
            continue

        df = read_csv_file(file_path)

        if df is None:
            print("Skipped unreadable file:", file_path)
            continue

        text_cols = find_text_columns(df)

        if not text_cols:
            print("Skipped no text column:", file_path, list(df.columns))
            continue

        # Case 1: file has a label column, like hard_cases.csv.
        # This has priority over filename-based labels.
        has_label_column = "label" in df.columns or "label_name" in df.columns

        if has_label_column:
            labeled_files_count += 1
            print("Loaded labeled file:", file_path, "| Rows:",
                  len(df), "| Text cols:", text_cols)

            for _, row in df.iterrows():
                parts = []

                for col in text_cols:
                    value = row.get(col)

                    if pd.notna(value):
                        value = str(value).strip()

                        if value:
                            parts.append(value)

                text = clean_text(" ".join(parts))

                if len(text) < 10:
                    continue

                parsed_label = parse_label_from_column(row)

                if parsed_label is None:
                    continue

                if parsed_label == 2:
                    # Binary model cannot learn NEEDS_REVIEW as a separate class.
                    # We skip it and keep the app logic responsible for Needs Review.
                    skipped_review_count += 1
                    if SKIP_NEEDS_REVIEW_IN_BINARY_TRAINING:
                        continue

                if parsed_label not in [0, 1]:
                    continue

                lang = detect_language(text)

                rows.append({
                    "text": text,
                    "label": parsed_label,
                    "language": lang,
                    "source_file": os.path.basename(file_path),
                    "is_hard_case": "hard_cases" in os.path.basename(file_path).lower()
                })

            continue

        # Case 2: old files without label column.
        # Infer label from the filename.
        label = infer_label_from_filename(file_path)

        if label is None:
            print("Skipped unknown label file:", file_path)
            continue

        print("Loaded:", file_path, "| Label:", label,
              "| Rows:", len(df), "| Text cols:", text_cols)

        for _, row in df.iterrows():
            parts = []

            for col in text_cols:
                value = row.get(col)

                if pd.notna(value):
                    value = str(value).strip()

                    if value:
                        parts.append(value)

            text = " ".join(parts)
            text = clean_text(text)

            if len(text) < 30:
                continue

            lang = detect_language(text)

            rows.append({
                "text": text,
                "label": label,
                "language": lang,
                "source_file": os.path.basename(file_path),
                "is_hard_case": False
            })

    print()
    print("Labeled files detected:", labeled_files_count)
    print("Skipped NEEDS_REVIEW rows because this is binary training:",
          skipped_review_count)

    return pd.DataFrame(rows)


def sample_with_priority(df, n, random_state=42):
    """
    Keep hard_cases.csv examples whenever possible, then fill the rest randomly.
    This prevents hard fake cases from being lost during balancing.
    """
    if n <= 0:
        return df.iloc[0:0]

    if "is_hard_case" not in df.columns:
        return df.sample(n=min(n, len(df)), random_state=random_state)

    priority_df = df[df["is_hard_case"] == True]
    normal_df = df[df["is_hard_case"] != True]

    if len(priority_df) >= n:
        return priority_df.sample(n=n, random_state=random_state)

    remaining_n = n - len(priority_df)

    if len(normal_df) == 0:
        return priority_df

    normal_sample = normal_df.sample(
        n=min(remaining_n, len(normal_df)),
        random_state=random_state
    )

    return pd.concat([priority_df, normal_sample], ignore_index=True)


def generate_extra_hard_fake_rows():
    """Extra high-impact fake claims that commonly confuse the model."""
    examples = [
        "قرار أممي بإلغاء العملات الورقية نهائيًا بحلول العام القادم",
        "سيتم إيقاف العمل بالعملات الورقية نهائيًا ابتداءً من العام القادم",
        "أعلنت منظمة دولية إلغاء جميع العملات الورقية حول العالم خلال سنة واحدة",
        "البنك المركزي أعلن إلغاء الدينار واستبداله بعملة رقمية خلال أسبوع",
        "ناسا تعلن انحراف الأرض عن مسارها واقترابها من الشمس",
        "ناسا تؤكد أن الأرض ستخرج من مدارها خلال أيام",
        "كوكب اليابان يعلن اختراع حبة دواء تغني عن النوم نهائيًا",
        "تم تعيين شخص غير معروف قائدًا للجيش الأمريكي دون إعلان رسمي",
        "تم تعيين أحمد عبد السلام حميدي قائدًا للجيش الأمريكي",
        "تم تعيين حمزة عبوني رئيسًا لفلسطين",
        "وفاة الرئيس الأمريكي دون صدور بيان رسمي من البيت الأبيض",
        "الرئيس الأمريكي استخدم السلاح النووي ضد إيران في الحرب الأخيرة",
        "الحكومة قررت قطع الإنترنت عن جميع المواطنين يوم الانتخابات",
        "البنوك ستجمد حسابات المواطنين خلال 48 ساعة إذا لم يحدثوا بياناتهم عبر رابط متداول",
        "واتساب سيغلق حسابات المستخدمين الذين لا يرسلون رسالة تأكيد إلى عشرة أشخاص",
        "تناول الثوم يوميًا يمنع الإصابة بجميع الفيروسات بنسبة 100%",
        "القهوة مع الليمون تحرق دهون البطن خلال يوم واحد دون رياضة أو حمية",
        "شركة ميتا ستجعل جميع صور المستخدمين ملكًا لها إذا لم ينشروا رسالة اعتراض",
        "المدارس ستتعطل لمدة شهر كامل بسبب موجة حر دون إعلان رسمي",
        "تم إلغاء الانتخابات القادمة وتأجيلها خمس سنوات بقرار مفاجئ",
        "The United Nations decided to cancel all paper currencies next year",
        "NASA announced that Earth has left its orbit and is moving toward the sun",
        "The US president used nuclear weapons against Iran in the latest war",
        "An unknown person was appointed commander of the United States Army",
        "WhatsApp will delete accounts that do not forward a verification message",
        "Banks will freeze accounts unless users update information through a shared link",
        "Lemon coffee burns belly fat in one day without exercise",
        "Eating garlic prevents all viruses with 100 percent effectiveness"
    ]

    rows = []
    for text in examples:
        rows.append({
            "text": clean_text(text),
            "label": 1,
            "language": detect_language(text),
            "source_file": "extra_hard_fake_generated",
            "is_hard_case": True
        })

    return pd.DataFrame(rows)


class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1
    }


def prepare_data():
    df = load_all_files()

    print()
    print("Loaded rows:")
    print(df.shape)

    if df.empty:
        raise ValueError(
            "No valid data found. Check CSV files and column names.")

    # Remove duplicate or near-identical duplicated rows BEFORE splitting.
    # This helps prevent data leakage between train/validation/test.
    before_dedup = len(df)
    df["text_norm"] = (
        df["text"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.drop_duplicates(subset=["text_norm"]).drop(columns=["text_norm"])
    after_dedup = len(df)

    print()
    print("Duplicates removed before split:", before_dedup - after_dedup)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print()
    print("Before balancing:")
    print(df.groupby(["language", "label"]).size())

    print()
    print("Hard cases loaded:")
    if "is_hard_case" in df.columns:
        print(df[df["is_hard_case"] == True].groupby(
            ["language", "label"]).size())
    else:
        print("No is_hard_case column found.")

    balanced_parts = []

    for lang in sorted(df["language"].unique()):
        lang_df = df[df["language"] == lang]

        real_df = lang_df[lang_df["label"] == 0]
        fake_df = lang_df[lang_df["label"] == 1]

        if len(real_df) == 0 or len(fake_df) == 0:
            print(f"Skipped language {lang} because one label is missing.")
            continue

        size = min(len(real_df), len(fake_df), MAX_PER_LABEL_PER_LANGUAGE)

        real_sample = sample_with_priority(real_df, size, random_state=42)
        fake_sample = sample_with_priority(fake_df, size, random_state=42)

        balanced_parts.append(real_sample)
        balanced_parts.append(fake_sample)

    if not balanced_parts:
        raise ValueError(
            "Could not create balanced dataset. Check labels and languages.")

    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    balanced_df = balanced_df.sample(
        frac=1, random_state=42).reset_index(drop=True)

    print()
    print("After balancing:")
    print(balanced_df.groupby(["language", "label"]).size())
    print("Total:", len(balanced_df))

    print()
    print("Hard cases after balancing:")
    if "is_hard_case" in balanced_df.columns:
        print(balanced_df[balanced_df["is_hard_case"] ==
              True].groupby(["language", "label"]).size())

    train_df, temp_df = train_test_split(
        balanced_df,
        test_size=0.30,
        random_state=42,
        stratify=balanced_df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    # Oversample hard FAKE cases in training only.
    # This makes the model learn high-impact false claims strongly without polluting validation/test.
    hard_fake_train = train_df[
        (train_df.get("is_hard_case", False) == True) &
        (train_df["label"] == 1)
    ].copy()

    extra_parts = []

    if len(hard_fake_train) > 0:
        repeated_hard_fake = pd.concat(
            [hard_fake_train] * HARD_FAKE_REPEAT_IN_TRAIN,
            ignore_index=True
        )
        extra_parts.append(repeated_hard_fake)
        print()
        print("Oversampled hard fake rows added to TRAIN:",
              len(repeated_hard_fake))

    if ADD_SYNTHETIC_HARD_FAKE:
        synthetic_hard_fake = generate_extra_hard_fake_rows()
        synthetic_hard_fake = pd.concat(
            [synthetic_hard_fake] * SYNTHETIC_HARD_FAKE_REPEAT,
            ignore_index=True
        )
        extra_parts.append(synthetic_hard_fake)
        print("Synthetic hard fake rows added to TRAIN:",
              len(synthetic_hard_fake))

    if extra_parts:
        train_df = pd.concat([train_df] + extra_parts, ignore_index=True)
        train_df = train_df.sample(
            frac=1, random_state=42).reset_index(drop=True)

    print()
    print("Final train distribution after hard-case boosting:")
    print(train_df.groupby(["language", "label"]).size())

    train_df.to_csv(os.path.join(PREPARED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PREPARED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PREPARED_DIR, "test.csv"), index=False)
    balanced_df.to_csv(os.path.join(
        PREPARED_DIR, "clean_dataset.csv"), index=False)

    print()
    print("Saved prepared data:")
    print("prepared_data/train.csv", len(train_df))
    print("prepared_data/val.csv", len(val_df))
    print("prepared_data/test.csv", len(test_df))
    print("prepared_data/clean_dataset.csv", len(balanced_df))

    return train_df, val_df, test_df


def build_training_arguments(args_dict):
    try:
        return TrainingArguments(
            **args_dict,
            eval_strategy="epoch"
        )
    except TypeError:
        try:
            return TrainingArguments(
                **args_dict,
                evaluation_strategy="epoch"
            )
        except TypeError:
            args_dict.pop("warmup_ratio", None)
            return TrainingArguments(
                **args_dict,
                evaluation_strategy="epoch"
            )


def save_training_plots(trainer, output_dir):
    """Save training/validation curves from HuggingFace Trainer logs."""
    logs = trainer.state.log_history

    train_epochs = []
    train_losses = []

    eval_epochs = []
    eval_losses = []
    eval_accuracies = []
    eval_f1_scores = []

    for log in logs:
        if "loss" in log and "epoch" in log and "eval_loss" not in log:
            try:
                train_epochs.append(float(log["epoch"]))
                train_losses.append(float(log["loss"]))
            except Exception:
                pass

        if "eval_loss" in log and "epoch" in log:
            try:
                eval_epochs.append(float(log["epoch"]))
                eval_losses.append(float(log["eval_loss"]))

                if log.get("eval_accuracy") is not None:
                    eval_accuracies.append(float(log.get("eval_accuracy")))
                else:
                    eval_accuracies.append(None)

                if log.get("eval_f1_fake") is not None:
                    eval_f1_scores.append(float(log.get("eval_f1_fake")))
                else:
                    eval_f1_scores.append(None)

            except Exception:
                pass

    os.makedirs(output_dir, exist_ok=True)

    if train_losses:
        plt.figure()
        plt.plot(train_epochs, train_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Training Loss Over Time")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "training_loss.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    if eval_losses:
        plt.figure()
        plt.plot(eval_epochs, eval_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss Over Epochs")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "validation_loss.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    valid_acc_points = [(e, a) for e, a in zip(
        eval_epochs, eval_accuracies) if a is not None]
    if valid_acc_points:
        xs, ys = zip(*valid_acc_points)
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Validation Accuracy Over Epochs")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "validation_accuracy.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    valid_f1_points = [(e, f) for e, f in zip(
        eval_epochs, eval_f1_scores) if f is not None]
    if valid_f1_points:
        xs, ys = zip(*valid_f1_points)
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation F1-score Fake")
        plt.title("Validation F1-score Over Epochs")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "validation_f1_fake.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    print()
    print("Training plots saved to:", output_dir)
    print("- training_loss.png")
    print("- validation_loss.png")
    print("- validation_accuracy.png")
    print("- validation_f1_fake.png")


def save_confusion_matrix_plot(cm, output_dir):
    """Save a simple confusion matrix figure for the final test results."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks([0, 1], ["REAL", "FAKE"])
    plt.yticks([0, 1], ["REAL", "FAKE"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print("Confusion matrix plot saved to:", os.path.join(
        output_dir, "confusion_matrix.png"))


def train_model(train_df, val_df, test_df):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    print()
    print("Using device:", device_name)

    train_df = train_df[["text", "label"]].dropna()
    val_df = val_df[["text", "label"]].dropna()
    test_df = test_df[["text", "label"]].dropna()

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    print()
    print("Dataset sizes:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    print()
    print("Train label distribution:")
    print(train_df["label"].value_counts())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = NewsDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = NewsDataset(val_df, tokenizer, MAX_LEN)
    test_dataset = NewsDataset(test_df, tokenizer, MAX_LEN)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={
            0: "REAL",
            1: "FAKE"
        },
        label2id={
            "REAL": 0,
            "FAKE": 1
        }
    )

    args_dict = {
        "output_dir": OUTPUT_DIR,
        "save_strategy": "no",
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": 0.1,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": NUM_EPOCHS,
        "weight_decay": 0.01,
        "logging_steps": 50,
        "load_best_model_at_end": False,
        "metric_for_best_model": "f1_fake",
        "greater_is_better": True,
        "save_total_limit": 1,
        "report_to": "none",
        "fp16": torch.cuda.is_available()
    }

    training_args = build_training_arguments(args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print()
    print("Starting training...")
    trainer.train()

    save_training_plots(trainer, OUTPUT_DIR)

    print()
    print("Evaluating on test set...")
    test_output = trainer.predict(test_dataset)

    logits = test_output.predictions
    true_labels = test_output.label_ids
    pred_labels = np.argmax(logits, axis=1)

    accuracy = accuracy_score(true_labels, pred_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        average="binary",
        zero_division=0
    )

    cm = confusion_matrix(true_labels, pred_labels)
    save_confusion_matrix_plot(cm, OUTPUT_DIR)

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=["REAL", "FAKE"],
        zero_division=0
    )

    results = {
        "test_accuracy": accuracy,
        "test_precision_fake": precision,
        "test_recall_fake": recall,
        "test_f1_fake": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    print()
    print("Test Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("Model saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    train_df, val_df, test_df = prepare_data()
    train_model(train_df, val_df, test_df)
