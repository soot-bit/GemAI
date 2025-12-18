  
<div align="center">
<h1 style="display: flex; align-items: center; gap: 10px;">
  <img src="https://4vector.com/i/free-vector-diamant-diamond_100183_Diamant_diamond.png" alt="Diamond icon" style="width: 40px; height: 40px;">
  GemAI
</h1>
<p>An end-to-end solution for diamond price prediction using deep learning on tabular data.</p>

  <img src="https://img.shields.io/badge/Predictor-Diamonds-blueviolet" />
  <img src="https://img.shields.io/badge/Model-TabNet-red" />
  <img src="https://img.shields.io/badge/API-FastAPI-green" />
  <img src="https://img.shields.io/badge/Optimisation-Optuna-yellow" />
  <img src="https://img.shields.io/badge/License-MIT-blue" />
</div>


## 🔮 The Story Behind GemAI

I built GemAI out of pure curiosity, I wanted to play with Google's TabNet model and see how deep learning could handle structured tabular data (something we usually throw XGBoost at). Turns out, it's pretty good ...

*Why diamonds?* Well, they're kind of a big deal where I'm from - not just sparkly rocks but serious business. This project let me combine my interest in deep learning with something culturally relevant.

### The Jist

- **TabNet is powerful but quirky:** - The training process feels different from traditional models. You can't just throw your usual tricks at it and expect magic.
  
- **tuning:** - Finding the right hyperparameters was like trying to find the perfect diamond cut - lots of trial and error, but worth it when you get it right!

- **GPU Acceleartion:**  Night and day difference ...

I'd say TabNet  is on par with good old traditional ML for this kind of data (eg.  XGBoost or Random Forests ...) It's another tool in the toolbox with its own strengths. I guess the real selling point is X-AI which i have to agree with the feature importance.




![Web App Preview](./app/web_app.png)

## Features

- **High Accuracy Predictions**: Trained on 50k+ diamond samples
- Automated tuning with Optuna
- **Low latency Inference**: prediction latency
- **Interactive Web UI**
- **RESTful APIs** 
- **Production-ready API with FastAPI**

## 🧠 Tech Stack

| Component       | Technology                                                                 |
|-----------------|----------------------------------------------------------------------------|
| **Core Model**  | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) [TabNet](https://github.com/dreamquark-ai/tabnet) |
| **Optimization**| ![Optuna](https://img.shields.io/badge/Optuna-2C3E50?logo=optuna&logoColor=white) |
| **API**         | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) |
| **Frontend**    | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black) ![Jinja2](https://img.shields.io/badge/Jinja2-B41717?logo=jinja&logoColor=white) |
| **Packaging**   | ![Pydantic](https://img.shields.io/badge/Pydantic-92000F?logo=pydantic&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) |
| **Data Source** | ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=white) Diamonds Dataset |

## 📊 Exploratory Data Analysis

Comprehensive analysis of diamond features and their relationships:

[![Open EDA Notebook](./notebooks/pairplot.png)](./notebooks/EDA.ipynb)

*Click image to view full analysis notebook*

## 📁 Project Structure

The project has been refactored for better organization, maintainability, and a CLI-first workflow.

```
/
├── app/                  # FastAPI application for serving the model
├── configs/              # All configuration files (e.g., config.toml)
├── data/                 # Data storage
│   ├── raw/              # Raw, immutable datasets
│   └── processed/        # Cleaned and processed datasets
├── models/               # Saved model artifacts and preprocessing mappings
├── notebooks/            # Jupyter notebooks for exploration and analysis
├── src/                  # Source code for the project
│   └── GemAI/            # Main Python package for GemAI
│       ├── config.py     # Centralized configuration loading
│       ├── data.py       # Data loading utilities
│       ├── main.py       # Main Command Line Interface (CLI) entry point
│       ├── utils.py      # General utility functions (e.g., logging)
│       └── models/       # Package for model-related modules
│           ├── autogluon.py  # AutoGluon training logic
│           └── tabnet.py     # TabNet training and tuning logic
├── tests/                # Unit and integration tests
├── pyproject.toml        # Project metadata and dependencies (managed by uv)
└── README.md             # This file
```

## ⚙️ Workflow and Usage

This project is managed via a centralized command-line interface (CLI). The recommended workflow is outlined below.

### 1. Setup

First, clone the repository and install the required dependencies using `uv`.

```bash
# Clone the repository
git clone https://github.com/your-username/GemAI.git
cd GemAI

# Install main and development dependencies using uv
# (Ensure uv is installed: pip install uv)
uv install .[dev]
```

### 2. Data Preprocessing

Generate the cleaned dataset from the raw `diamonds.csv` file. This script will perform all the cleaning steps from the EDA notebook and save the output to `data/processed/clean_ds.plk`.

```bash
uv run python -m src.GemAI.main process-data
```

### 3. Hyperparameter Tuning (Optional but Recommended)

Run Optuna to find the best hyperparameters for the TabNet model. The results will be saved back into `configs/config.toml` for the training step.

```bash
uv run python -m src.GemAI.main tune tabnet
```

### 4. Model Training

Train the model using the hyperparameters from your configuration file.

```bash
# To train the TabNet model (uses parameters from tune step)
uv run python -m src.GemAI.main train tabnet

# To train the AutoGluon model
uv run python -m src.GemAI.main train autogluon
```
The trained TabNet model and its preprocessing mappings will be saved in the `models/tabnet/` directory.

### 5. Serving the Prediction API

Launch the FastAPI application to serve predictions. The API uses the latest trained TabNet model from the previous step.

```bash
uv run python -m src.GemAI.main serve
```
- The API will be available at `http://localhost:8000`.
- The interactive web GUI is at `/`.
- The interactive API documentation (Swagger UI) is at `/docs`.

### 6. Testing the API

The project includes a set of API tests. These tests require the server to be running in a separate terminal.

```bash
# In one terminal, start the server:
uv run python -m src.GemAI.main serve

# In another terminal, run pytest:
uv run pytest
```

### API Request Example

You can interact with the running API using `curl` or any HTTP client.

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "carat": 0.75,
    "cut": "Ideal",
    "color": "D",
    "clarity": "IF",
    "depth": 62.1,
    "table": 57,
    "x": 5.71,
    "y": 5.73,
    "z": 3.55
}'
```
**Expected Response:**
```json
{
  "price_bwp": 12345.67
}
```
*(The price is an example and will vary based on the trained model.)*


