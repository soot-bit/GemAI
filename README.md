


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

- **GPU Acceleartion:**  Night and day difference from CPU *chef's kiss*.

I'd say TabNet  is on par with good old traditional ML for this kind of data (eg.  XGBoost or Random Forests ...) It's another tool in the toolbox with its own strengths. I guess the real selling point is X-AI which i have to agree with the feature importance.




![Web App Preview](./app/web_app.png)

## Features

- **High Accuracy Predictions**: Trained on 50k+ diamond samples
- Automated tuning with Optuna
- **Low lantency Inference**: prediction latency
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

## ⚙️ Use:

*use uv preferably but pip still works*, [uv documentation](https://docs.astral.sh/uv/)

```bash
# Clone 
git clone https://github.com/yourusername/GemAI.git
cd GemAI

# in a  virtual environment
pip install uv 
uv install .
```

**Run  Server**
```bash
uvicorn app.main:app --reload
```
Visit `http://localhost:8000` in your browser

**API Request**
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
or use web GUI

## 📚 Documentation

### API Endpoints
| Endpoint       | Method | Description               | Request Body                                   |
|----------------|--------|---------------------------|-----------------------------------------------|
| `/predict`     | POST   | Get price prediction      | JSON with diamond features                    |
| `/`            | GET    | Web interface             | -                                             |
| `/docs`        | GET    | Interactive API docs      | -                                             |

### Input Parameters
```json
{
  "carat": 0.75,
  "cut": "Ideal",
  "color": "D",
  "clarity": "IF",
  "depth": 62.1,
  "table": 57,
  "x": 5.71,
  "y": 5.73,
  "z": 3.55
}

```

