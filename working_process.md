# DecisionLens Working Process & Analytical Methodology

This document explains the mathematical workflow of how the application evaluates branch efficiency and outlines the specific features available within the interactive Business Intelligence dashboard.

---

## 1. How Efficiency is Calculated

The system employs **Data Envelopment Analysis (DEA)**, a rigorous operations research technique, specifically using the **CCR (Charnes-Cooper-Rhodes) Input-Oriented model** solved via Linear Programming.

### The Variables
- **Inputs (Resources consumed):** Staff count, Operating Cost.
- **Outputs (Results produced):** Deposits 2016, Deposit Growth, Average Deposits.

### The Mathematical Process
1. **Normalization:** The model first normalizes all input and output columns across the dataset so that large monetary values (like deposits) do not mathematically overshadow smaller integer metrics (like staff members).
2. **Linear Programming:** For every single branch (Decision-Making Unit or DMU), the system formulates a targeted linear programming problem using the `PuLP` library. 
3. **Weight Optimization:** The solver computationally determines the most favorable possible weights for that specific branch's inputs and outputs. It attempts to maximize the branch's efficiency ratio (weighted outputs divided by weighted inputs).
4. **The Constraint:** The fundamental strict constraint of DEA is that if these optimal weights were applied to any other branch in the dataset, no branch could achieve an efficiency score greater than 1.0.

### The Scoring and Classification
- The final **Efficiency Score** falls strictly between `0.0` and `1.0`.
- **Efficient (Score ≥ 0.9999):** The branch pushes the maximum possible output for the lowest possible input. It physically sits on the "Efficiency Frontier" and serves as a benchmark for others.
- **Inefficient (Score < 0.9999):** The branch is mathematically proven to be wasting resources. The model has identified other real-world branches in the dataset that achieved the same or better output using strictly fewer inputs. 

---

## 2. Dashboard Interface & Features

The Streamlit desktop application (`app.py`) transforms these complex mathematical results into an intuitive, multi-tab interface for business analysts and managers. 

### 📊 Tab 1: Overview
A high-level view of the entire branch network.
- **Fleet Health Summary:** Global KPI trackers showing network-wide average efficiency, and the exact count of efficient vs. inefficient branches.
- **Efficiency Leaderboard:** A scalable bar chart indexing the top-performing branches.
- **Score Distribution:** A histogram showing the frequency of efficiency scores so managers can see if the network is heavily skewed toward inefficiency.
- **Efficiency Frontier Scatter Plot:** Plots inputs (Staff) against outputs (Deposits). Efficient branches hug the outer "frontier" edge of the graph, visually trapping the inefficient branches behind them.

### 🔬 Tab 2: Branch Analysis
A focused, deep-dive evaluation of one selected branch.
- **Deep-Dive Metrics:** Displays the exact efficiency score, network rank, and variances vs the fleet average.
- **Radar Comparison Chart:** A web-like overlapping visual overlaying the selected branch's attributes against the fleet average and the #1 top-performing branch.
- **Performance Diagnosis:** Auto-generates flags identifying localized weaknesses, such as a poorer-than-average deposit-to-staff ratio or an inflated cost-to-deposit ratio.

### 🗺️ Tab 3: Geographic Map
- **Geospatial Plotting:** An interactive map plotting all branches across the United States. Branches are color-coded in a traffic-light scheme (Green for Highly Efficient, Red for Severely Inefficient) allowing executives to instantly spot regional performance trends.

### 📋 Tab 4: Data Explorer
- **Interactive Data Grid:** A sortable, formatted table containing the raw performance metrics and calculated status labels for every single branch in the fleet, useful for exporting or deep raw-number auditing.

### 🔮 Tab 5: What-If Simulator
A real-time predictive modeling tool.
- **Slack Analysis & Recommendations:** Translates mathematical shortfalls into plain-text business directives (e.g., "Reduce Operating Cost by 12%").
- **Interactive Sliders:** Users can dynamically "lay off staff" or "increase deposit goals" via UI sliders.
- **Real-Time Recalculation:** Triggers an isolated, micro-DEA linear programming execution that immediately predicts the new simulated efficiency score based on the adjusted slider variables.

### 📐 Tab 6: Scenario Comparison
- **Strategic Memory:** As users test theories in the Simulator, they can save them to the Scenario Store. This tab lines up the saved "What-If" scenarios in a grouped bar chart, allowing management to visually compare original vs. projected efficiency to pick the best real-world intervention strategy.
