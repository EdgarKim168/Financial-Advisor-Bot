# Financial-Advisor-Bot
Final year Project - Financial Advisor Bot for Foreign Exchange

## Make sure to download the requirements.txt before running the streamlit_app. 
use this command to run streamlit_app
```bash
streamlit run streamlit_app.py
```

evaluate_advisor.py is used for backtesting and evaluation. This script runs the advisor logic on historical forex data and generates performance graphs, metrics, and evaluation results. It provides the charts and figures that appear in the report’s Evaluation chapter.

to get the graph and data that I added in my report, use this command 

```bash
python evaluate_advisor.py --pair "EUR/USD" --interval 1h --period 60d --risk-frac 0.01

