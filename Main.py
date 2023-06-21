import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import statistics as sta
import matplotlib.colors as mcolors
import base64
import matplotlib.pyplot as plt
import numpy as np

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
# Apply CSS styling to the tables
table_style = """
    <style>
    table {
        color: #333;
        font-family: Arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
        border: 1px solid #ccc;
    }

    table th {
        background-color: #f4f4f4;
        text-align: left;
        padding: 8px;
    }

    table td {
        padding: 8px;
    }

    table tr:nth-child(even) {
        background-color: #f9f9f9;
    }

    table tr:hover {
        background-color: #f1f1f1;
    }

    </style>
"""
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    but_summary,but_stats,but_charts,but_streaks,but_drawdown,but_datatable = st.columns(6)
    cost_col,capital_col=st.columns(2)
    data = pd.read_csv(uploaded_file)
    def color_negative_red(value):
        if isinstance(value, str):
            value = float(value)
        else:
            value = float(value.item())
        if value < 0:
            return 'background-color: #ff6961'
        elif value > 0:
            return 'background-color: #77dd77'
        else:
            return ''
    data = data.sort_values('ExitTime')
    #setting the cost
    with cost_col:
        cost = st.number_input('Insert the Cost')
    data['EntryPrice']=data['EntryPrice']*(1-(cost/100))
    data['ExitPrice']=data['ExitPrice']*(1+(cost/100))
    data['Pnl']=(data['EntryPrice']-data['ExitPrice'])*data['Quantity']*data['PositionStatus']*-1

    with capital_col:
        capital=st.number_input('Insert Capital')

    #running basics
    data['ExitTime'] = pd.to_datetime(data['ExitTime'])
    data['year'] = data['ExitTime'].dt.year
    data['month_name'] = data['ExitTime'].dt.month_name()
    data = data.sort_values('ExitTime')
    # Extract month and year from the date
    data['month_year'] = data['ExitTime'].dt.to_period('M')
    data['day_of_week'] = data['ExitTime'].dt.day_name()
    options1,options2=st.columns(2)
    with options1:
        options_year = st.multiselect('Years', data['year'].unique(), key='years_multiselect')
    with options2:
        options_days = st.multiselect('Days', day_order, key='days_multiselect')
    if options_year!=[]:
        data = data[data['ExitTime'].dt.year.isin(options_year)].reset_index(drop=True)
    if options_days!=[]:
        data = data[data['ExitTime'].dt.day_name().isin(options_days)].reset_index(drop=True)
    # =============================================================================STREAKS=========================================================================================
    # Calculate the streaks
    data['is_win'] = data['Pnl'] > 0
    data['is_loss'] = data['Pnl'] < 0

    # Calculate the streak ID for winning and losing streaks
    data['win_streak_id'] = (data['is_win'] != data['is_win'].shift()).cumsum()
    data['loss_streak_id'] = (data['is_loss'] != data['is_loss'].shift()).cumsum()

    win_streaks = data[data['is_win']].groupby('win_streak_id').agg(
        Days=('ExitTime', 'count'),
        Start=('ExitTime', 'first'),
        End=('ExitTime', 'last'),
        Profit=('Pnl', 'sum')
    ).nlargest(5, 'Profit').reset_index(drop=True)
    win_streaks = win_streaks[['Days', 'Start', 'End', 'Profit']]
    
    # Calculate the streak details for losing streaks
    loss_streaks = data[data['is_loss']].groupby('loss_streak_id').agg(
        Days=('ExitTime', 'count'),
        Start=('ExitTime', 'first'),
        End=('ExitTime', 'last'),
        Loss=('Pnl', 'sum')
    ).nsmallest(5, 'Loss').reset_index(drop=True)
    loss_streaks = loss_streaks[['Days', 'Start', 'End', 'Loss']]

#=========================================================================DRAWDOWN=======================================================================================
    # Calculate drawdownimport pandas as pd

    data['cumulative_pnl'] = data['Pnl'].cumsum()
    data['previous_peak'] = data['cumulative_pnl'].cummax()
    data['drawdown'] = data['cumulative_pnl'] - data['previous_peak']
    drawdown_periods = []
    current_drawdown = None
    for i in range(len(data)):
        if current_drawdown is None:
            if data['drawdown'][i] < 0:
                current_drawdown = {'Start Date': data['ExitTime'][i], 'Max Date': data['ExitTime'][i], 'End Date': None, 'Drawdown': float('inf')}
        else:
            if data['drawdown'][i] >= 0:
                current_drawdown['End Date'] = data['ExitTime'][i-1]
                if current_drawdown['Start Date'] != current_drawdown['End Date']:
                    drawdown_periods.append(current_drawdown)
                    current_drawdown = None
            else:
                if data['drawdown'][i] < current_drawdown['Drawdown']:
                    current_drawdown['Drawdown'] = data['drawdown'][i]
                    current_drawdown['Max Date'] = data['ExitTime'][i]

    if current_drawdown is not None and current_drawdown['Start Date'] != current_drawdown['End Date']:
        current_drawdown['End Date'] = data['ExitTime'][len(data)-1]
        drawdown_periods.append(current_drawdown)

    drawdown_df = pd.DataFrame(drawdown_periods)
    drawdown_graph=px.bar(drawdown_df,y='Drawdown',title="Drawdown")
#======================================================================================Stats======================================================================================

# ----------------------------------------------------------------------------------Monthly Breakup----------------------------------------------------------------------------------
    monthly_pnl = data.groupby('month_year')['Pnl'].sum().reset_index().astype(str)

# ----------------------------------------------------------------------------------Daily Breakup----------------------------------------------------------------------------------
    # Group the data by year and day of the week and calculate the sum of profit for each combination
    daywise_breakup = data.groupby(['year', 'day_of_week'])['Pnl'].sum().unstack().reindex(day_order,axis=1)

# --------------------------------------------------------------------------------Ratios--------------------------------------------------------------------------------

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
    data['month'] = pd.to_datetime(data['ExitTime']).dt.to_period('M')
    average_monthly_profit = data.groupby('month')['Pnl'].mean().mean()

    # Calculate the average profit on win days
    average_profit_win_days = data[data['Pnl'] > 0]['Pnl'].mean()

    # Calculate the average loss on loss days
    average_loss_loss_days = data[data['Pnl'] < 0]['Pnl'].mean()
    avg_yearly_profit = data.groupby(data['ExitTime'].dt.year)['Pnl'].sum().mean()
    median_monthly_profit = data.groupby(data['ExitTime'].dt.to_period('M'))['Pnl'].sum().median()
    # Calculate Average Weekly Profit
    avg_weekly_profit = data.groupby(data['ExitTime'].dt.to_period('W'))['Pnl'].sum().mean()

    # Calculate Average Trades Per Day
    avg_trades_per_day = data.groupby(data['ExitTime'].dt.date)['Key'].count().mean()
    data["Month"] = data["ExitTime"].dt.month
    data['Date']=data['ExitTime'].dt.date
    max_drawdown=abs(data['drawdown'].min())
    if capital==0:
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
    data['std']=pd.to_numeric(data['Pnl'])/capital
    std=data['std'].values.tolist()
    # stdev=sta.pstdev(std)
    stdev = np.std(std)
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
# --------------------------------------------------------------------------------minimum_pnl--------------------------------------------------------------------------------
    month_order = ['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December']
    data['monthly_pnl_unstyled'] = data.groupby(['year', 'month_name'])['Pnl'].cumsum()
    minimum_pnl = data.groupby(['year', 'month_name'])['monthly_pnl_unstyled'].min().unstack().fillna(0).reindex(month_order, axis=1)

# --------------------------------------------------------------------------------monthly_trades--------------------------------------------------------------------------------
    # Group the trades by month and calculate the win rate
    monthly_trades_overview = data.groupby(['year', 'month_name']).apply(lambda x: (x['Pnl'] > 0).sum() / len(x) * 100).unstack().fillna(0).reindex(month_order, axis=1)

# ==========================================================================but_summary======================================================================================
    
    #----------------------------------------------------------------------------------quaterly breakup----------------------------------------------------------------------------------

    data['quarter'] = data['ExitTime'].dt.quarter

    # Calculate the total PNL for each quarter and year
    quarterly_pnl = data.groupby(['year', 'quarter'])['Pnl'].sum().unstack().fillna(0)
    quarterly_pnl['Net Pnl'] = quarterly_pnl.sum(axis=1)
    total_pnl = quarterly_pnl.sum(axis=1)
    quarterly_pnl_percent = (quarterly_pnl / capital) * 100

    bar_fig_quarterly = go.Figure()
    for col in quarterly_pnl.columns:
        bar_fig_quarterly.add_trace(go.Bar(x=quarterly_pnl.index, y=quarterly_pnl[col], name=f"Q{col}"))

    bar_fig_quarterly.update_layout(
        title='Quarterly PNL',
        xaxis_title='Year',
        yaxis_title='PNL'
    )

#----------------------------------------------------------------------------------cumulative pnl----------------------------------------------------------------------------------

    data['pnl_cumulative'] = data['Pnl'].cumsum()
    # Create the area plot for cumulative PNL
    area_fig = go.Figure(data=go.Scatter(x=data['ExitTime'], y=data['pnl_cumulative'], fill='tozeroy'))

    # Set the layout for the area plot
    area_fig.update_layout(
        title='Cumulative PNL',
        xaxis_title='Date',
        yaxis_title='Cumulative PNL'
    )


# ======================================================================================but_charts======================================================================================

    data['pnl_cumulative'] = data['Pnl'].cumsum()
    # Create the area plot for cumulative PNL
    area_fig = go.Figure(data=go.Scatter(x=data['ExitTime'], y=data['pnl_cumulative'], fill='tozeroy'))

    # Set the layout for the area plot
    area_fig.update_layout(
        title='Cumulative PNL',
        xaxis_title='Date',
        yaxis_title='Cumulative PNL'
    )

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
    data['Daily PNL'] = data.groupby(data['ExitTime'].dt.date)['Pnl'].cumsum()
    daily_fig = px.bar(data, x='Date', y='Daily PNL', title='Daily PNL Cumulative')

    # Calculate the cumulative PNL on a weekly basis
    data['Week'] = data['ExitTime'].dt.to_period('W').astype(str)
    data['Weekly PNL'] = data.groupby('Week')['Pnl'].cumsum()
    weekly_fig = px.bar(data, x='Week', y='Weekly PNL', title='Weekly PNL Cumulative')


    #monthly trades
    monthly_trades = data.groupby('month_year').size().reset_index(name='Number of Trades').astype(str)

    # Create a bar graph for monthly number of trades using Plotly
    bar_fig_trades = go.Figure(data=go.Bar(x=monthly_trades['month_year'], y=monthly_trades['Number of Trades']))
    bar_fig_trades.update_layout(
        title='Monthly Number of Trades',
        xaxis_title='Month',
        yaxis_title='Number of Trades'
    )

# =========================================================DATA TABLE=====================================================================================================================================================
        # Apply color formatting to PNL values

    # Apply color formatting to the entire DataFrame
    selected_headers = ['Key', 'ExitTime', 'EntryPrice', 'ExitPrice', 'Pnl', 'PositionStatus', 'Quantity', 'Symbol']

    # Subset the data with selected headers
    subset_data = data[selected_headers]
    styled_data_table = subset_data.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x).applymap(lambda x: x.rstrip('0').rstrip('.') if isinstance(x, str) else x).style.applymap(color_negative_red, subset=['Pnl'])

    # Calculate the total PNL for each month and year
    monthly_pnl_unstyled = data.groupby(['year', 'month_name'])['Pnl'].sum().unstack()
    monthly_pnl_unstyled['Net Pnl'] = monthly_pnl_unstyled.sum(axis=1)
    month_order = ['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December','Net Pnl']
    monthly_pnl = monthly_pnl_unstyled.reindex(month_order, axis=1).applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x).applymap(lambda x: x.rstrip('0').rstrip('.') if isinstance(x, str) else x).style.applymap(color_negative_red)


# ====================================================running the code===============================================================================================
    def display(x):
        if x==3:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Display the area plot
                st.plotly_chart(area_fig)
                st.divider()
                st.plotly_chart(bar_fig_monthly)
                st.divider()
                st.plotly_chart(weekly_fig)
                st.divider()
                st.plotly_chart(daily_fig)
                st.divider()
                st.plotly_chart(bar_fig_trades)
        if x==4:
            col1, col2 = st.columns(2)

            with col1:
                st.write("Top 5 Maximum Winning Streaks (Moneywise):")
                st.markdown(table_style, unsafe_allow_html=True)
                st.table(win_streaks.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x).applymap(lambda x: x.rstrip('0').rstrip('.') if isinstance(x, str) else x).style.applymap(lambda x: 'color: #77dd77')) 
                st.write("Top 5 Maximum Losing Streaks (Moneywise):")
                st.markdown(table_style, unsafe_allow_html=True)
                st.table(loss_streaks.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x).applymap(lambda x: x.rstrip('0').rstrip('.') if isinstance(x, str) else x).style.applymap(lambda x: 'color: #ff6961'))      

            with col2:
                st.write("Top 5 Longest Winning Streaks (Timewise):")
                st.markdown(table_style, unsafe_allow_html=True)
                st.table(win_streaks.nlargest(5, 'Days').reset_index(drop=True).applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x).applymap(lambda x: x.rstrip('0').rstrip('.') if isinstance(x, str) else x).style.applymap(lambda x: 'color: #77dd77')) 

                st.write("Top 5 Longest Losing Streaks (Timewise):")
                st.markdown(table_style, unsafe_allow_html=True)
                st.table(loss_streaks.nlargest(5, 'Days').reset_index(drop=True).applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x).applymap(lambda x: x.rstrip('0').rstrip('.') if isinstance(x, str) else x).style.applymap(lambda x: 'color: #ff6961'))
        if x==6:
            st.markdown(table_style, unsafe_allow_html=True)
            st.table(styled_data_table)
        if x==2:
            st.header('Monthly Breakup')
            # Display the monthly PNL breakup
            st.table(monthly_pnl)
            st.divider()
            s1,s2,s3=st.columns(3)
            with s1:
                # Display the table with custom styling
                st.markdown(table_style, unsafe_allow_html=True)
                st.table(pd.DataFrame.from_dict(statistics, orient='index', columns=['Value']))

            with s2:
                # Display the table with custom styling
                st.markdown(table_style, unsafe_allow_html=True)
                st.table(pd.DataFrame.from_dict(Stats2, orient='index', columns=['Value']))

            with s3:
                # Display the table with custom styling
                st.markdown(table_style, unsafe_allow_html=True)
                st.table(pd.DataFrame.from_dict(Ratios, orient='index', columns=['Value']))

            st.divider()
            st.header('Day- Wise Breakup')
            # Display the day=wise breakup as a table
            st.markdown(table_style, unsafe_allow_html=True)
            st.table(daywise_breakup.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x).applymap(lambda x: x.rstrip('0').rstrip('.') if isinstance(x, str) else x).style.applymap(color_negative_red))
            st.divider()
            st.markdown(table_style, unsafe_allow_html=True)
            st.subheader('Monthly Win Rate')
            st.table(monthly_trades_overview.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x).applymap(lambda x: x.rstrip('0').rstrip('.') if isinstance(x, str) else x).style.applymap(color_negative_red))
            st.divider()
            st.subheader('Minimum PNL')
            st.markdown(table_style, unsafe_allow_html=True)
            st.table(minimum_pnl.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x).applymap(lambda x: x.rstrip('0').rstrip('.') if isinstance(x, str) else x).style.applymap(color_negative_red))
        if x==1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.subheader("Quarterly PNL Breakup (Absolute Values)")
                st.markdown(table_style, unsafe_allow_html=True)
                st.table(quarterly_pnl.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x).applymap(lambda x: x.rstrip('0').rstrip('.') if isinstance(x, str) else x).style.applymap(color_negative_red))
                st.subheader("Quarterly PNL Breakup (Percentages)")
                st.markdown(table_style, unsafe_allow_html=True)
                st.table(quarterly_pnl_percent.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x).applymap(lambda x: x.rstrip('0').rstrip('.') if isinstance(x, str) else x).style.applymap(color_negative_red))
                st.divider()
            with col2:
                st.plotly_chart(area_fig)
                st.divider() 
            with col2:
                st.header('Quarterly Bar Chart')
                st.plotly_chart(bar_fig_quarterly)
        if x==5:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.plotly_chart(drawdown_graph)
            st.markdown(table_style, unsafe_allow_html=True)
            st.table(drawdown_df.sort_values(by='Drawdown').reset_index(drop=True).style.applymap(lambda x: 'color: #ff6961',subset=['Drawdown']))
    with but_charts:
        if st.button('Charts'):
            x=3
    with but_streaks:
        if st.button('Streaks'):
            x=4
    with but_stats:
        if st.button('Stats'):
            x=2
    with but_summary:
        if st.button('Summary'):
            x=1
    with but_datatable:
        if st.button('Data Table'):
            x=6
    with but_drawdown:
        if st.button('Drawdown'):
            x=5

    display(x)
