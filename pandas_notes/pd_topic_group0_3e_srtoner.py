# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# + [Pandas Query](#Pandas-Query) 
# + [Time Series](#Time-Series) 
# + [Window Functions](#Window-Functions) 

# ## Pandas Query ##
# 
# ### pd. query ##
# 
# ###### Name: Anandkumar Patel
# ###### Email: patelana@umich.edu
# ###### Unique ID: patelana

# ### Arguments and Output
# 
# **Arguments** 
# 
# * expression (expr) 
# * inplace (default = False) 
#     * Do you want to operate directly on the dataframe or create new one
# * kwargs (keyword arguments)
# 
# **Returns** 
# * Dataframe from provided query

# ## Why
# 
# * Similar to an SQL query 
# * Can help you filter data by querying
# * Returns a subset of the DataFrame
# * loc and iloc can be used to query either rows or columns

# ## Query Syntax
# 
# * yourdataframe.query(expression, inplace = True/False

# ## Code Example

# In[2]:


import pandas as pd
df = pd.DataFrame({'A': range(1, 6),
                   'B': range(10, 0, -2),
                   'C C': range(10, 5, -1)})
print(df)

print('Below is the results of the query')

print(df.query('A > B'))





# ## Time Series
# **Name: Lu Qin**
# UM email: qinlu@umich.edu

# ### Overview
#  - Data times
#  - Time Frequency
#  - Time zone

# ### Import

import datetime
import pandas as pd
import numpy as np

# ### Datetime
#  - Parsing time series information from various sources and formats

# +
dti = pd.to_datetime(
    ["20/10/2021", 
     np.datetime64("2021-10-20"), 
     datetime.datetime(2021, 10, 20)]
)

dti
# -

# ### Time frequency
# - Generate sequences of fixed-frequency dates and time spans
# - Resampling or converting a time series to a particular frequency

# #### Generate

# +
dti = pd.date_range("2021-10-20", periods=2, freq="H")

dti
# -

# #### convert

# +
idx = pd.date_range("2021-10-20", periods=3, freq="H")
ts = pd.Series(range(len(idx)), index=idx)

ts
# -

# #### resample

ts.resample("2H").mean()

# ### Timezone
#  - Manipulating and converting date times with timezone information
#  - `tz_localize()`
#  - `tz_convert()`

# +
dti = dti.tz_localize("UTC")
dti

dti.tz_convert("US/Pacific")
# -

# ## Window Functions ##
# **Name: Stephen Toner** \
# UM email: srtoner@umich.edu
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web

# Of the many funcitons in Pandas, one which is particularly useful for time
# series analysis is the window function. It lets us apply some aggregation 
# function over a specified lookback period on a rolling basis throughout the
# time series. This is particularly useful for financial analsyis of equity
# returns, so we will compute some financial metrics for Amazon stock using
# this techinique.

# Our first step is to import our data for Amazon ("AMZN") 
# over a healthy time horizon:

# +
amzn_data = web.DataReader("AMZN", 
                           data_source = 'yahoo', 
                           start = "2016-10-01", 
                           end = "2021-10-01")

amzn_data.head()
# -


# While the column labels are largely self-explanatory, two important notes
# should be made:
# * The adjusted close represents the closing price after all is said and done
# after the trading session ends; this may represent changes due to accounts 
# being settled / netted against each other, or from adjustments to financial
# reporting statements.
# * One reason for our choice in AMZN stock rather than others is that AMZN
# has not had a stock split in the last 20 years; for this reason we do not
# need to concern ourselves with adjusting for the issuance of new shares like
# we would for TSLA, AAPL, or other companies with large
# market capitalization.

# Getting back to Pandas, we have three main functions that allow us to
# perform Window operations:
# * `df.shift()`: Not technically a window operation, but helpful for
# computing calculations with offsets in time series
# * `rolling`: For a given fixed lookback period, tells us the 
# aggregation metric (mean, avg, std dev)
# * `expanding`: Similar to `rolling`, but the lookback period is not fixed. 
# Helpful when we want to have a variable lookback period such as "month to 
# date" returns

# Two metrics that are often of interest to investors are the returns of an
# asset and the volume of shares traded. Returns are either calculated on
# a simple basis:
# $$ R_s = P_1/P_0 -1$$
# or a log basis:
# $$ R_l = \log (P_1 / P_2) $$
# Simple returns are more useful when aggregating returns across multiple 
# assets, while Log returns are more flexible when looking at returns across 
# time. As we are just looking at AMZN, we will calculate the log returns
# using the `shift` function:

# +
amzn_data["l_returns"] = np.log(amzn_data["Adj Close"]/
                                amzn_data["Adj Close"].shift(1))


plt.title("Log Returns of AMZN")
plt.plot(amzn_data['l_returns'])

# -

# For the latter, we see that the
# volume of AMZN stock traded is quite noisy:

plt.title("Daily Trading Volume of AMZN")   
plt.plot(amzn_data['Volume'])

# If we want to get a better picture of the trends, we can always take a
# moving average of the last 5 days (last full set of trading days):

amzn_data["vol_5dma"] = amzn_data["Volume"].rolling(window = 5).mean()
plt.title("Daily Trading Volume of AMZN")   
plt.plot(amzn_data['vol_5dma'])

# When we apply this to a price metric, we can identify some technical patterns
# such as when the 15 or 50 day moving average crosses the 100 or 200 day
# moving average (known as the golden cross, by those who believe in it).

# +
amzn_data["ma_15"] = amzn_data["Adj Close"].rolling(window = 15).mean()
amzn_data["ma_100"] = amzn_data["Adj Close"].rolling(window = 100).mean()

fig1 = plt.figure()
plt.plot(amzn_data["ma_15"])
plt.plot(amzn_data["ma_100"])
plt.title("15 Day MA vs. 100 Day MA")

# We can then use the `shift()` method to identify which dates have 
# golden crosses

gc_days = (amzn_data.eval("ma_15 > ma_100") & 
               amzn_data.shift(1).eval("ma_15 <= ma_100"))

gc_prices = amzn_data["ma_15"][gc_days]


fig2 = plt.figure()
plt.plot(amzn_data["Adj Close"], color = "black")
plt.scatter( x= gc_prices.index, 
                y = gc_prices[:],
                marker = "+", 
                color = "gold" 
                )

plt.title("Golden Crosses & Adj Close")
# -

# The last feature that Pandas offers is a the `expanding` window function, 
# which calculates a metric over a time frame that grows with each additional 
# period. This is particularly useful for backtesting financial metrics
# as indicators of changes in equity prices: because one must be careful not
# to apply information from the future when performing backtesting, the 
# `expanding` functionality helps ensure we only use information up until the 
# given point in time. Below, we use the expanding function to plot cumulative
# return of AMZN over the time horizon.

# +
def calc_total_return(x):
    """    
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.log(x[-1] / x[0]) 


amzn_data["Total Returns"] = (amzn_data["Adj Close"]
                              .expanding()
                              .apply(calc_total_return))

fig3 = plt.figure()
ax5 = fig3.add_subplot(111)
ax5 = plt.plot(amzn_data["Total Returns"])
plt.title("Cumulative Log Returns for AMZN")
# -


