# Aerospace Backorder Prediction

## Setup
1. Install Python 3.8 or higher
2. Install required libraries:
```
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly
```

3. Place these 4 CSV files in the same folder as the scripts:
   - `parts_master.csv`
   - `supply_chain_history.csv`
   - `purchase_orders.csv`
   - `quality_incidents.csv`

## Running the Analysis

This generates all figures and prints results to the console:

```
python MSE433_analysis.py
```

Figures are saved to a `figures/` folder in the same directory.

## Running the Dashboard

```
streamlit run dashboard.py
```

This opens an interactive dashboard in your browser where you can:
- Select a week to view predictions for
- Adjust the expediting budget and criticality weights
- View the recommended expedite list and download it as CSV
- See model performance charts and sensitivity analysis
