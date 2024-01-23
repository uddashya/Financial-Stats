# Finstats

## Overview

Finstats is a Streamlit-based application designed for financial data analysis, particularly for backtesting trading strategies. It provides a comprehensive set of features including profit and loss calculations, statistical analysis, and visual representations of trading data.

## Features

- **Data Processing**: Process and analyze trading data from uploaded CSV files.
- **Statistical Analysis**: Compute key trading metrics such as overall profit, win/loss percentage, average monthly profit, and more.
- **Visualizations**: Generate various charts and graphs including cumulative P&L, monthly P&L, win/loss streaks, and drawdown analysis.
- **Customization**: Filter data by years and days, and adjust cost settings for more precise analysis.
- **User Interface**: Easy-to-use web interface built with Streamlit.

## Installation

To run Multyfi Backtester, you need to have Python installed on your system along with the following libraries:
- Streamlit
- Pandas
- Plotly
- NumPy
- Matplotlib
- Pillow
- Requests

Use the following command to install the required libraries:

```bash
pip install streamlit pandas plotly numpy matplotlib Pillow requests
```

## Usage

1. **Start the Application**: Run the application using Streamlit.
    ```bash
    streamlit run finstats_main.py
    ```
2. **Upload Data**: Upload your trading data in CSV format.
3. **Set Parameters**: Customize the cost settings and select specific years or days for analysis.
4. **Analyze Data**: View statistical summaries, streaks, drawdowns, and various charts based on the trading data.

## Demo video

https://drive.google.com/file/d/1Ts8yaB_MQdYNzzWTwEgJ4Qfi0OKrGj5Q/view

## Contributing

Contributions to Finstats are welcome. Please ensure to update tests as appropriate.

