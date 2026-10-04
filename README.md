# FactGuard – AI-Powered Misinformation Detection System

FactGuard is a Computer Science graduation project developed to help users evaluate the credibility of Arabic and English news content.

The system combines transformer-based text classification with external evidence retrieval and Natural Language Inference (NLI) to provide an evidence-aware misinformation verification process.

Rather than relying only on a classification model, FactGuard can search for external evidence, compare that evidence with the submitted content, and use the combined results to produce a final credibility decision.

---

## Key Features

- Arabic and English news analysis
- Transformer-based misinformation classification
- External evidence retrieval from trusted sources
- Natural Language Inference (NLI) for evidence comparison
- Credibility classification such as Real, Fake, or Needs Review
- Flask-based backend
- REST API for prediction requests
- SQLite-based data storage
- Web-based user interface
- Chrome extension integration
- Trusted-source configuration

---

## System Architecture

FactGuard follows a multi-stage verification pipeline:

```text
User Input
    ↓
Text Preprocessing
    ↓
Transformer Classifier
    ↓
External Evidence Search
    ↓
Natural Language Inference (NLI)
    ↓
Final Credibility Decision
```

The classifier provides an initial prediction, while the evidence-retrieval and NLI stages provide additional context before the system produces its final result.

---

## Tech Stack

### Backend

- Python
- Flask
- REST APIs
- SQLite

### AI / NLP

- PyTorch
- Hugging Face Transformers
- XLM-RoBERTa
- Natural Language Inference (NLI)

### Data & Training

- Pandas
- Hugging Face Datasets
- Scikit-learn

### Frontend & Integration

- HTML
- CSS
- JavaScript
- Chrome Extension

### External Integration

- Tavily API for external information retrieval

---

## Model Performance

During the original graduation project evaluation, the misinformation classification model achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | 98.18% |
| Fake Precision | 98.73% |
| Recall | 97.61% |
| F1 Score | 98.16% |
| Test Samples | 1,592 |

These results were obtained during the original project training and evaluation process and should be interpreted within the context of the project's dataset and methodology.

---

## Chrome Extension

FactGuard includes a Chrome extension that integrates the verification functionality into the user's browsing experience.

The extension communicates with the Flask backend through the `/predict` endpoint and can display a credibility signal while viewing content on Facebook.

This demonstrates how the FactGuard backend can be consumed by an external client through its API.

---

## Project Structure

```text
factguard-misinformation-detection/
│
├── extension/              # Chrome extension
├── static/                 # CSS, JavaScript, and static assets
├── templates/              # Flask HTML templates
│
├── app.py                  # Flask backend and API
├── train.py                # Model training pipeline
├── trusted_websites.json   # Trusted-source configuration
│
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
└── README.md
```

Large datasets, prepared training data, trained model weights, generated files, local databases, and environment secrets are intentionally excluded from this public repository.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hamzaesam144-web/factguard-misinformation-detection.git
cd factguard-misinformation-detection
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example`.

Example:

```env
TVLY_API=your_tavily_api_key_here
```

Do not commit your real `.env` file or API credentials to GitHub.

---

## Running the Application

> **Important:** The trained model weights are not included in this repository because of their large file size.

To run the complete misinformation detection pipeline locally, the required trained model artifacts must first be generated using the training pipeline or placed in the model directory expected by the application.

The repository contains the source code for the Flask backend, REST API, web interface, training pipeline, evidence-verification logic, and Chrome extension.

Once the required model artifacts are available:

```bash
python app.py
```

---

## Model Training

The model training pipeline is provided in:

```text
train.py
```

The project uses XLM-RoBERTa for multilingual misinformation classification.

The original datasets, prepared datasets, and trained model artifacts are intentionally excluded from this public repository because of their size.

The reported evaluation results were obtained during the original graduation project training and evaluation process.

---

## REST API

The Flask backend exposes endpoints used by the web application and Chrome extension.

The primary prediction endpoint is:

```text
POST /predict
```

The endpoint accepts content for analysis and processes it through the FactGuard verification pipeline.

The API architecture allows other clients, such as the included Chrome extension, to communicate with the misinformation-detection backend.

---

## Evidence Verification

FactGuard goes beyond standalone text classification by incorporating external evidence into the verification process.

The general process is:

```text
Submitted Content
        ↓
Initial Classification
        ↓
External Evidence Retrieval
        ↓
Evidence Comparison using NLI
        ↓
Final Credibility Assessment
```

This approach was designed to make the result more informative than relying exclusively on the classifier prediction.

---

## Trusted Sources

The repository contains:

```text
trusted_websites.json
```

This file is used as part of the project's trusted-source configuration for the evidence-verification process.

---

## Environment Variables

Sensitive values are stored using environment variables rather than directly inside the source code.

Example configuration:

```env
TVLY_API=your_tavily_api_key_here
```

The real `.env` file is excluded from Git through `.gitignore`.

Only `.env.example`, which contains placeholder values, is included in the repository.

---

## Files Excluded from the Repository

The following types of files are intentionally excluded:

```text
.env
Dataset/
prepared_data/
factguard_mmbert_small/
factguard_xlm_roberta/
*.pth
*.pt
*.bin
*.safetensors
*.db
__pycache__/
venv/
```

These include sensitive configuration, large datasets, trained model artifacts, generated files, local databases, caches, and virtual environments.

---

## Project Context

FactGuard was developed as a **Computer Science graduation project at the University of Jordan**.

The project provided practical experience with:

- Backend development using Python and Flask
- REST API development
- Database-backed web applications
- Integration of NLP models into a working software system
- External API integration
- Browser-extension integration
- Arabic and English text processing
- Model training and evaluation workflows

---

## Repository Purpose

This repository provides the source code and technical structure of the FactGuard graduation project.

Because the original trained models and datasets are several gigabytes in size, they are not distributed directly through this GitHub repository.

The repository is primarily intended to demonstrate the application's architecture, backend implementation, API integration, training pipeline, browser extension, and overall project design.

---

## Disclaimer

FactGuard is an academic project and should not be considered a professional fact-checking authority.

Automated misinformation detection systems can produce incorrect or incomplete results. Important claims should always be verified using reliable and independent sources.

---

## Author

**Hamza Khalil**

Computer Science Graduate – University of Jordan

GitHub: https://github.com/hamzaesam144-web

LinkedIn: https://www.linkedin.com/in/hamza-khalil-2597653b8