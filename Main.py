import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import statistics as sta
import matplotlib.colors as mcolors
import base64
base="dark"

st.set_page_config(

    page_title="Multyfi Backtester",
    page_icon="🗓️",
    layout="wide",
)
image_path = '/Users/uddashyakumar/Desktop/Screenshot 2023-06-16 at 14.27.04.png'  # Replace with the path to your image file
with open(image_path, 'rb') as image_file:
    image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

image_data_uri = f'data:image/png;base64,{image_base64}'
caption = 'Backtest Engine'
width = 500  # Specify the desired width in pixels
st.image(image_data_uri,width=width)
x=0
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    but_summary,but_stats,but_charts,but_streaks,but_datatable = st.columns(5)
# ==========================================================================CHARTS=======================================================================================
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    def color_negative_red(value):
        if value < 0:
            return 'background-color: #ff6961'
        elif value > 0:
            return 'background-color: #77dd77'
        else:
            return ''

    # Convert the date column to datetime format
    data['date'] = pd.to_datetime(data['date'])
    data['pnl_cumulative'] = data['Pnl'].cumsum()
    # Create the area plot for cumulative PNL
    area_fig = go.Figure(data=go.Scatter(x=data['date'], y=data['pnl_cumulative'], fill='tozeroy'))

    # Set the layout for the area plot
    area_fig.update_layout(
        title='Cumulative PNL',
        xaxis_title='Date',
        yaxis_title='Cumulative PNL'
    )

    # Extract month and year from the date
    data['month_year'] = data['date'].dt.to_period('M')

    # Calculate the total PNL for each month
    monthly_pnl = data.groupby('month_year')['Pnl'].sum().reset_index().astype(str)

    # Create a bar graph for monthly PNL using Plotly
    bar_fig_monthly = go.Figure(data=go.Bar(x=monthly_pnl['month_year'], y=monthly_pnl['Pnl']))
    bar_fig_monthly.update_layout(
        title='Monthly PNL',
        xaxis_title='Month',
        yaxis_title='PNL'
    )

    # Calculate the cumulative PNL on a daily basis
    data['pnl_cumsum_daily'] = data.groupby(data['date'].dt.date)['Pnl'].cumsum()
    daily_fig = px.bar(data, x='date', y='pnl_cumsum_daily', title='Daily PNL Cumulative')

    # Calculate the cumulative PNL on a weekly basis
    data['week'] = data['date'].dt.to_period('W').astype(str)
    data['pnl_cumsum_weekly'] = data.groupby('week')['Pnl'].cumsum()
    weekly_fig = px.bar(data, x='week', y='pnl_cumsum_weekly', title='Weekly PNL Cumulative')


    #monthly trades
    monthly_trades = data.groupby('month_year').size().reset_index(name='Number of Trades').astype(str)

    # Create a bar graph for monthly number of trades using Plotly
    bar_fig_trades = go.Figure(data=go.Bar(x=monthly_trades['month_year'], y=monthly_trades['Number of Trades']))
    bar_fig_trades.update_layout(
        title='Monthly Number of Trades',
        xaxis_title='Month',
        yaxis_title='Number of Trades'
    )
    # =============================================================================STREAKS=========================================================================================
    # Sort the data by date
    data = data.sort_values('date')

    # Calculate the streaks
    data['is_win'] = data['Pnl'] > 0
    data['is_loss'] = data['Pnl'] < 0

    # Calculate the streak ID for winning and losing streaks
    data['win_streak_id'] = (data['is_win'] != data['is_win'].shift()).cumsum()
    data['loss_streak_id'] = (data['is_loss'] != data['is_loss'].shift()).cumsum()

    # Calculate the streak details for winning streaks
    win_streaks = data[data['is_win']].groupby('win_streak_id').agg(
        Days=('date', 'count'),
        Start=('date', 'first'),
        End=('date', 'last'),
        Profit=('Pnl', 'sum')
    ).nlargest(5, 'Profit')

    # Calculate the streak details for losing streaks
    loss_streaks = data[data['is_loss']].groupby('loss_streak_id').agg(
        Days=('date', 'count'),
        Start=('date', 'first'),
        End=('date', 'last'),
        Loss=('Pnl', 'sum')
    ).nsmallest(5, 'Loss')
    # ============================================================MONTHLY BREAKUP====================================================================================================================================================
    # Convert the date column to datetime format
    daily_pnl = data.groupby('date')['Pnl'].sum().reset_index()
    # Extract month and year from the date
    data['year'] = data['date'].dt.year
    data['month_name'] = data['date'].dt.month_name()

    # =========================================================DATA TABLE=====================================================================================================================================================
        # Apply color formatting to PNL values

    # Apply color formatting to the entire DataFrame
    selected_headers = ['Key', 'ExitTime', 'EntryPrice', 'ExitPrice', 'Pnl', 'PositionStatus', 'Quantity', 'Symbol']

    # Subset the data with selected headers
    subset_data = data[selected_headers]
    styled_data_table = subset_data.style.applymap(color_negative_red, subset=['Pnl'])

    # Calculate the total PNL for each month and year
    monthly_pnl_unstyled = data.groupby(['year', 'month_name'])['Pnl'].sum().unstack()
    monthly_pnl = monthly_pnl_unstyled.style.applymap(color_negative_red)


    # ============================================================QUATERLY BREAKUP===============================================================================================================================================
    data['quarter'] = data['date'].dt.quarter

    # Calculate the total PNL for each quarter and year
    quarterly_pnl = data.groupby(['year', 'quarter'])['Pnl'].sum().unstack().fillna(0)
    total_pnl = quarterly_pnl.sum(axis=1)
    quarterly_pnl_percent = (quarterly_pnl / np.array(total_pnl)[:, None]) * 100
    bar_fig_quarterly = go.Figure()
    for col in quarterly_pnl.columns:
        bar_fig_quarterly.add_trace(go.Bar(x=quarterly_pnl.index, y=quarterly_pnl[col], name=f"Q{col}"))

    bar_fig_quarterly.update_layout(
        title='Quarterly PNL',
        xaxis_title='Year',
        yaxis_title='PNL'
    )
    #=======================================================================STATS=========================================================================================

    # Calculate the overall profit
    overall_profit = data['Pnl'].sum()

    # Calculate the average day profit
    average_day_profit = data['Pnl'].mean()

    # Calculate the maximum profit
    max_profit = data['Pnl'].max()

    # Calculate the maximum loss
    max_loss = data['Pnl'].min()

    # Calculate the win percentage (days)
    win_percentage = (data[data['Pnl'] > 0].shape[0] / data.shape[0]) * 100

    # Calculate the loss percentage (days)
    loss_percentage = (data[data['Pnl'] < 0].shape[0] / data.shape[0]) * 100

    # Calculate the average monthly profit
    data['month'] = pd.to_datetime(data['date']).dt.to_period('M')
    average_monthly_profit = data.groupby('month')['Pnl'].mean().mean()

    # Calculate the average profit on win days
    average_profit_win_days = data[data['Pnl'] > 0]['Pnl'].mean()

    # Calculate the average loss on loss days
    average_loss_loss_days = data[data['Pnl'] < 0]['Pnl'].mean()
    avg_yearly_profit = data.groupby(data['date'].dt.year)['Pnl'].sum().mean()
    median_monthly_profit = data.groupby(data['date'].dt.to_period('M'))['Pnl'].sum().median()
    # Calculate Average Weekly Profit
    avg_weekly_profit = data.groupby(data['date'].dt.to_period('W'))['Pnl'].sum().mean()

    # Calculate Average Trades Per Day
    avg_trades_per_day = data.groupby(data['date'].dt.date)['Key'].count().mean()
    #=========================================================================DRAWDOWN=======================================================================================


    # Calculate drawdown
    data['cumulative_pnl'] = data['Pnl'].cumsum()
    data['previous_peak'] = data['cumulative_pnl'].cummax()
    data['drawdown'] = data['cumulative_pnl'] - data['previous_peak']
# ===========================================================================RATIOS======================================================================================================
    data["ExitTime"] = pd.to_datetime(data["ExitTime"])
    data["Month"] = data["ExitTime"].dt.month
    data['Date']=data['date'].dt.date
    max_drawdown=abs(data['drawdown'].min())
    max_entries_day = data["Date"].value_counts().max()
    capital=(150000*abs(max_entries_day)+max_drawdown)*1.2
    data['NAv']=data["cumulative_pnl"].add(capital)
    data['NAv'] = pd.to_numeric(data['NAv'], errors='coerce')
    ddpercentage=(max_drawdown/capital)
    number_of_years=data['year'].nunique()
    cagr=(data['NAv'].iloc[-1]/capital)**(1/number_of_years)-1
    calmar=(cagr*capital)/max_drawdown
    average_points=overall_profit/(data['cumulative_pnl'].count())/data['Quantity'].iloc[5]
    roi_percentage=(overall_profit/capital)*100
    yearly_roi_percentage=(roi_percentage/number_of_years)
    data['std']=data['Pnl']/capital
    std=data['std'].values.tolist()
    stdev=sta.pstdev(std)
    sharpe_ratio=(cagr-.02)/stdev
    Ratios={
        'Maximum Drawdown':round(max_drawdown,2),
        'Drawdown Percentage':str(round(ddpercentage*100,2))+' %',
        'Cagr':round(cagr*100,2),
        'Calmar':round(calmar,2),
        'Roi Percentage':str(round(roi_percentage,2))+' %',
        'Yearly Roi Percentage':str(round(yearly_roi_percentage,2))+' %',
        'Sharpe Ratio':round(sharpe_ratio,2)
    }
    statistics = {
        'Overall Profit': overall_profit,
        'Average Day Profit': average_day_profit,
        'Avg Monthly Profit': average_monthly_profit,
        "Avg Yearly Profit": avg_yearly_profit,
        "Median Monthly Profit": median_monthly_profit,
        'Average points':round(average_points,2),
        "Avg Trades Per Day": avg_trades_per_day
    }
    Stats2={
        'Max Profit': max_profit,
        'Max Loss': max_loss,
        'Win% (Days)': win_percentage,
        'Loss% (Days)': loss_percentage,
        'Avg Profit On Win Days': average_profit_win_days,
        'Avg Loss On Loss Days': average_loss_loss_days,
        "Avg Weekly Profit": avg_weekly_profit,
    }
# =================================================================================daywise_breakup==========================================================================
    # Extract the day of the week from the date
    data['day_of_week'] = data['date'].dt.day_name()
    # Group the data by year and day of the week and calculate the sum of profit for each combination
    daywise_breakup = data.groupby(['year', 'day_of_week'])['Pnl'].sum().unstack()
    

# ====================================================running the code===============================================================================================
    def display(x):
        if x==1:
            # Display the area plot
            st.plotly_chart(area_fig)
            st.divider()
            st.plotly_chart(daily_fig)
            st.divider()
            st.plotly_chart(bar_fig_monthly)
            st.divider()
            st.plotly_chart(weekly_fig)
            st.divider()
            st.plotly_chart(bar_fig_trades)
        if x==2:
            col1, col2 = st.columns(2)

            with col1:
                st.write("Top 5 Maximum Winning Streaks (Moneywise):")
                st.dataframe(win_streaks.style.applymap(lambda x: 'color: #77dd77')) 
                st.write("Top 5 Maximum Losing Streaks (Moneywise):")
                st.dataframe(loss_streaks.nlargest(5, 'Loss').style.applymap(lambda x: 'color: #ff6961'))      

            with col2:
                st.write("Top 5 Longest Winning Streaks (Timewise):")
                st.dataframe(win_streaks.nlargest(5, 'Days').style.applymap(lambda x: 'color: #77dd77')) 

                st.write("Top 5 Longest Losing Streaks (Timewise):")
                st.dataframe(loss_streaks.nlargest(5, 'Days').style.applymap(lambda x: 'color: #ff6961'))
        if x==6:
            st.table(styled_data_table)
        if x==4:
            s1,s2,s3=st.columns(3)
            with s1:
                st.table(pd.DataFrame.from_dict(statistics, orient='index', columns=['Value']))
            with s2:
                st.table(pd.DataFrame.from_dict(Stats2, orient='index', columns=['Value']))
            with s3:
                st.table(pd.DataFrame.from_dict(Ratios, orient='index', columns=['Value']))
            st.divider()
            st.header('Monthly Breakup')
            # Display the monthly PNL breakup
            st.table(monthly_pnl)
            st.divider()
            st.header('Day- Wise Breakup')
            # Display the day=wise breakup as a table
            st.table(daywise_breakup.style.applymap(color_negative_red))
            st.divider()
        if x==5:
            sum1,sum2=st.columns(2)
            with sum1:
                st.subheader("Quarterly PNL Breakup (Absolute Values)")
                st.table(quarterly_pnl.style.applymap(color_negative_red))
            with sum2:
                st.subheader("Quarterly PNL Breakup (Percentages)")
                st.table(quarterly_pnl_percent.style.applymap(color_negative_red))
            st.divider()
            st.header('Cumulative Pnl Chart')
            st.plotly_chart(daily_fig)
            st.divider() 
            st.header('Quarterly Bar Chart')
            st.plotly_chart(bar_fig_quarterly)



    with but_charts:
        if st.button('Charts'):
            x=1
    with but_streaks:
        if st.button('Streaks'):
            x=2 
    with but_stats:
        if st.button('Stats'):
            x=4
    with but_summary:
        if st.button('Summary'):
            x=5
    with but_datatable:
        if st.button('Data Table'):
            x=6
    display(x)
    print(data['date'])
