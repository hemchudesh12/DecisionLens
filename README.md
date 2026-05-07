# 🔍 DecisionLens — Bank Branch Efficiency BI Platform

> A production-grade Business Intelligence platform that applies **Data Envelopment Analysis (DEA)** to evaluate, visualize, and simulate the operational efficiency of bank branches across an entire network.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![PuLP](https://img.shields.io/badge/PuLP-Linear%20Programming-orange)](https://coin-or.github.io/pulp/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-purple?logo=plotly)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

**DecisionLens** is an end-to-end BI solution built for banking operations. It ingests raw branch-level performance data, runs a rigorous mathematical optimization model (DEA), and surfaces the results through an interactive, multi-tab Streamlit dashboard — giving managers and analysts the power to:

- Instantly identify **efficient vs. inefficient** branches
- Benchmark underperformers against real top-performers in the same network
- Simulate **"What-If" scenarios** (e.g., *"What if we reduce staff by 10%?"*)
- Compare and save multiple strategic scenarios for executive review

---

## 🏗️ Architecture

```
bank_dea_dataset.csv
        │
        ├──▶ dea_model.py  ──▶  dea_results.csv
        │         │
        ├──▶ visualization.py ──┐
        │                       │
        ├──▶ simulation.py ─────┼──▶ app.py (Streamlit Dashboard)
        │                       │
        └──▶ intelligence.py ───┘
```

| Module | Role | Phase |
|---|---|---|
| `dea_model.py` | CCR Linear Programming Optimization Engine | Phase 2 |
| `visualization.py` | Plotly Chart & Map Generation | Phase 3 |
| `app.py` | Streamlit Frontend Dashboard | Phase 4 |
| `simulation.py` | Slack Analysis & What-If Simulator | Phase 5 |
| `intelligence.py` | AI Business Intelligence Consultant | Phase 6 |

---

## ⚙️ How It Works — The DEA Engine

DecisionLens uses the **CCR (Charnes-Cooper-Rhodes) Input-Oriented** DEA model, solved via Linear Programming.

### Inputs & Outputs

| Category | Variables |
|---|---|
| **Inputs** (Resources consumed) | Staff Count, Operating Cost |
| **Outputs** (Results produced) | Deposits 2016, Deposit Growth, Average Deposits |

### Mathematical Process

1. **Normalization** — All input/output columns are normalized so large monetary values don't overshadow smaller integer metrics.
2. **Per-Branch LP Formulation** — For every branch (Decision-Making Unit / DMU), a targeted linear program is constructed using `PuLP`.
3. **Weight Optimization** — The solver finds the most favorable weights for that branch's inputs/outputs, maximizing the efficiency ratio (weighted outputs ÷ weighted inputs).
4. **The DEA Constraint** — If those optimal weights were applied to any other branch, no branch can score above 1.0.

### Scoring & Classification

| Score | Status | Meaning |
|---|---|---|
| `≥ 0.9999` | ✅ **Efficient** | On the efficiency frontier; benchmark for others |
| `< 0.9999` | ❌ **Inefficient** | Mathematically proven resource waste; improvement target |

---

## 🖥️ Dashboard Features

### 📊 Tab 1 — Overview
- **Fleet Health KPIs:** Network-wide average efficiency, efficient vs. inefficient branch counts
- **Efficiency Leaderboard:** Ranked bar chart of top-performing branches
- **Score Distribution:** Histogram showing the network efficiency spread
- **Efficiency Frontier Scatter Plot:** Inputs vs. outputs, visually showing which branches hug the frontier

### 🔬 Tab 2 — Branch Analysis
- **Deep-Dive Metrics:** Exact efficiency score, network rank, variance vs. fleet average
- **Radar Comparison Chart:** Selected branch vs. fleet average vs. #1 top performer
- **Performance Diagnosis:** Auto-generated flags for localized weaknesses (e.g., poor deposit-to-staff ratio)

### 🗺️ Tab 3 — Geographic Map
- **Geospatial Plotting:** Interactive US map with color-coded branches
  - 🟢 Green = Highly Efficient
  - 🔴 Red = Severely Inefficient

### 📋 Tab 4 — Data Explorer
- Sortable, filterable data grid with all raw metrics and calculated status labels for every branch

### 🔮 Tab 5 — What-If Simulator
- **Slack Analysis & Recommendations:** Plain-English business directives (e.g., *"Reduce Operating Cost by 12%"*)
- **Interactive Sliders:** Dynamically adjust staff, costs, and deposit targets
- **Real-Time Recalculation:** Triggers a micro-DEA LP execution and instantly predicts the new efficiency score

### 📐 Tab 6 — Scenario Comparison
- Save multiple "What-If" runs to the Scenario Store
- Grouped bar chart comparing original vs. projected efficiency across all saved scenarios

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/hemchudesh12/DecisionLens.git
cd DecisionLens

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the DEA engine to generate results (first time only)
python dea_model.py

# 4. Launch the dashboard
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard framework |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computations |
| `plotly` | Interactive charts and maps |
| `scipy` | Scientific computing utilities |
| `pulp` | Linear Programming solver (DEA engine) |
| `matplotlib` | Supplementary plotting |
| `seaborn` | Statistical visualizations |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
DecisionLens/
├── app.py                  # Main Streamlit dashboard (entry point)
├── dea_model.py            # DEA CCR Linear Programming engine
├── visualization.py        # Plotly chart & map generation
├── simulation.py           # What-If simulator & slack analysis
├── intelligence.py         # AI business intelligence layer
├── data_preprocessing.py   # Data cleaning & normalization
├── bank_dea_dataset.csv    # Raw branch performance data
├── dea_results.csv         # Pre-computed DEA efficiency scores
├── requirements.txt        # Python dependencies
├── details.md              # Architecture documentation
├── working_process.md      # Methodology documentation
└── README.md               # This file
```

---

## 📈 Sample Insights Generated

- *"Branch #47 is operating at 72.3% efficiency. Reducing Operating Cost by ₹1.2M would push it above the efficiency frontier."*
- *"Branches in the Northeast cluster show a consistent pattern of overstaffing relative to deposit output."*
- *"Scenario B (10% staff reduction + 5% deposit growth target) projects a 14.7% efficiency gain over Scenario A."*

---

## 🧠 Intelligence Layer

The `intelligence.py` module acts as an **AI Business Consultant**, performing:

- **Root Cause Analysis** — Compares an underperforming branch against its top efficient peers
- **NLP-Style Insight Summaries** — Human-readable explanations of why a branch is underperforming
- **Sensitivity Analysis** — Recommends the highest ROI interventions (greatest efficiency gain per unit of change)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**hemchudesh12**  
[GitHub Profile](https://github.com/hemchudesh12)
