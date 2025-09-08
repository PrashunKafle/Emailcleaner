## Email Cleaner Using scikit-learn

A simple machine learning project that classifies emails as Spam, Promotions, or Important using scikit-learn and a Streamlit interface.

## Features
- Load and preprocess email datasets (`emails.csv`,`large_emails_dataset.csv`)
- Train a text classification model (Logistic Regression / Naive Bayes)
- Interactive web app built with Streamlit
- Suggests whether to keep or delete an email

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/PrashunKafle/Emailcleaner.git
   cd Emailcleaner


## Create and activate a virtual environment(Windows Example):
python -m venv .venv
.venv\Scripts\activate

## Install Dependencies 
pip install -r requirements.txt

## Use
streamlit run app.py

Then open the browser at http://localhost:8501

Emailcleaner/
─ app.py # Streamlit app (web interface)
─ Email_cleaner.py # Console-based classifier (CLI version)
─ emails.csv # Sample dataset (small)
─ large_emails_dataset.csv # Larger dataset for training/testing
─ requirements.txt # Project dependencies
─ README.md # Project documentation
─ .gitignore # Ignored files (virtual env, cache, etc.)
