# AI_Powered_Last_Mile_Delivery_Automation

> An end-to-end AI and LLMOps pipeline for Last Mile delivery company.

**Author:** Juan David Valderrama Artunduaga
**Python:** 3.13+

---

## Project Structure

```
├── params.yaml               # Training hyperparameters
├── schema.yaml               # Data schema definition
├── dvc.yaml                  # DVC pipeline DAG
├── main.py                   # Run all pipeline stages
├── app.py                    # FastAPI inference server
├── setup.py                  # Package metadata
├── requirements.txt          # Dependencies
├── src/AI_Powered_Last_Mile_Delivery_Automation/
│   ├── components/           # Reusable ML components
│   ├── config/               # ConfigurationManager
│   ├── logger/               # Logger configuration
│   ├── model/                # Model definitions
│   ├── prompts/              # Prompt templates
│   └── utils/                # Helper functions
├── static/                   # Static files for web UI
├── test/                     # Test files
├── .github/                  # GitHub Actions workflows
├── notebooks/trials.ipynb    # EDA & prototyping
├── data/
|   ├── raw/                  # Original data
|   ├── processed/            # Cleaned data
|   └── external/             # External datasets
└── templates/index.html      # Web UI
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Run training pipeline
python main.py

# 4. Serve predictions
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

## DVC Pipeline

```bash
dvc init
dvc repro        # Run the full pipeline
dvc dag          # Visualize stage dependencies
```
