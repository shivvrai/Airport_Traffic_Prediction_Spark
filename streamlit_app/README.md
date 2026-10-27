# ✈️ Airport Traffic Prediction — Streamlit Dashboard

An interactive web dashboard built with **Streamlit + Plotly + Scikit-Learn**, powered by the PySpark MLlib model results from the INT315 Cluster Computing project.

---

## 📁 Folder Structure
```
streamlit_app/
├── app.py              ← Main Streamlit dashboard
├── requirements.txt    ← Python dependencies
└── README.md           ← This file

(Parent folder — spark_Data — must contain:)
├── airport_traffic_2025.csv
├── colab_output.csv
└── model_evaluation_metrics.csv
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

The dashboard will open automatically at **http://localhost:8501**

---

## 📊 Dashboard Features

| Tab | What It Shows |
|:---|:---|
| **📊 Overview** | Monthly traffic trends, top 10 airports, day-of-week patterns, top countries |
| **🏆 Airport Explorer** | Select any of 331 airports → full year history, monthly breakdown, next-day forecast gauge |
| **🤖 Model Performance** | Model comparison table (MAE/RMSE/R²), feature importance donut chart, R² bar chart |
| **🗂️ Airport Clustering** | K-Means 3-tier clustering (Major / Medium / Small) with interactive scatter plot |

---

## 💼 Resume Point This Adds
> *"Deployed an interactive multi-tab analytics dashboard using Streamlit and Plotly, enabling real-time airport selection, next-day flight volume visualization, K-Means clustering of 331 airports into 3 operational tiers, and model performance benchmarking from trained PySpark MLlib pipeline outputs."*
