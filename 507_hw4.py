# -*- coding: utf-8 -*-
# # Stats 507 Problem Set 4
# *Work produced by Stephen Toner, Fall 2021*

# Import the following Modules:


import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import numpy as np
import pickle
import scipy.stats as stats
from IPython.core.display import display, HTML
import os.path as os
import warnings
import ci_funcs as ci

warnings.filterwarnings('ignore')

# # Question 0: Pandas Topics: Window Functions
#
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

#+ 
amzn_data["l_returns"] = np.log(amzn_data["Adj Close"]/
                                amzn_data["Adj Close"].shift(1))


plt.title("Log Returns of AMZN")
plt.plot(amzn_data['l_returns'])

# -

# For the latter, we see that the
# volume of AMZN stock traded is quite noisy:
   
    
# +
plt.title("Daily Trading Volume of AMZN")   
plt.plot(amzn_data['Volume'])
# -

# If we want to get a better picture of the trends, we can always take a
# moving average of the last 5 days (last full set of trading days):

# +    
amzn_data["vol_5dma"] = amzn_data["Volume"].rolling(window = 5).mean()
plt.title("Daily Trading Volume of AMZN")   
plt.plot(amzn_data['vol_5dma'])
# -

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
    x : nparray-like object
        Data of periodic financial returns

    Returns
    -------
    np.float64
        returns total log return of an equity
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


# # Question 1: NHANES Table 1

# ## Part a
# We make some minor adjustments to our code that combines NHANES survey data
# * Check to see if data is already downloaded
# * Include gender as a variable

# + 
def get_nhanes_data():
    """
    Effects: Fetches pre-determined dataset from the internet
    Returns
    -------
    demo_cohorts : pd.DataFrame
        Demographic data with cohort info
    oh_cohorts : pd.DataFrame
        Oral Health data with cohort, participant id

    """
    demo_cohorts = []
    oh_cohorts = []
    prefix = "https://wwwn.cdc.gov/Nchs/Nhanes/"
    demo_file = "/DEMO_"
    oh_file = "/OHXDEN_"
    ext = ".XPT"
    letters = ['G', 'H', 'I', 'J']
    j = 0
    for i in range(2011, 2019, 2): 
        year = str(i) + "-" + str(i + 1)
    
        demo_url = prefix + year + demo_file + letters[j] + ext
        oh_url = prefix + year + oh_file + letters[j] + ext
        j += 1    
        demo_new_df = pd.read_sas(demo_url)
        oh_new_df = pd.read_sas(oh_url)
    
        demo_new_df["Cohort"] = year
        oh_new_df["Cohort"] = year
        demo_cohorts.append(demo_new_df)
        oh_cohorts.append(oh_new_df)   
    
    return demo_cohorts, oh_cohorts

demo_columns = ["Cohort", 
                    "SEQN", 
                    "RIDAGEYR",
                    "RIAGENDR",
                    "RIDRETH3", 
                    "DMDEDUC2", 
                    "DMDMARTL",
                    "RIDSTATR", 
                    "SDMVPSU", 
                    "SDMVSTRA", 
                    "WTMEC2YR", 
                    "WTINT2YR"]

demo_labels = { 
                "SEQN" : "id", 
                "RIDAGEYR" : "age",
                "RIAGENDR" : "gender",
                "RIDRETH3" : "ethnicity", 
                "DMDEDUC2" : "education", 
                "DMDMARTL" : "marital_status",
                "RIDSTATR" : "exam_status", 
                "SDMVPSU" : "masked_var_unit", 
                "SDMVSTRA" : "pseudo_strat", 
                "WTMEC2YR" : "interview_weight", 
                "WTINT2YR" : "exam_weight"
                }

oh_columns = ["Cohort", "SEQN", "OHDDESTS"]

oh_labels = {"SEQN" : "id", "OHDDESTS" : "ohx_status"}

oh_skip_list = [1,16,17,32] # these numbers should be skipped for condition var

for i in range(1,33):
    oh_prefix = "OHX"
    count_suffix = "TC"
    cond_suffix = "CTC"
    i_str = str(i)
    if i < 10:
        i_str = "0"+i_str
    
    oh_columns.append(oh_prefix + i_str + count_suffix)
    oh_labels[oh_prefix + i_str + count_suffix] = "tooth_count" + i_str
   
    if(i not in oh_skip_list):
        oh_labels[oh_prefix + i_str + cond_suffix] = "tooth_cond" + i_str
        oh_columns.append(oh_prefix + i_str + cond_suffix)

file_list = ["nhanes_demo.pkl", "nhanes_dental.pkl"]

missing = [x for x in file_list if not os.exists(x)]

if missing:
    demo_cohorts, oh_cohorts = get_nhanes_data()
    
    demo_data = pd.concat(demo_cohorts)
    demo_data = demo_data[demo_columns]
    demo_data = demo_data.rename(columns=demo_labels)

    oh_data = pd.concat(oh_cohorts)
    oh_data = oh_data[oh_columns]
    oh_data = oh_data.rename(columns=oh_labels)
    
    demo_data["id"] = pd.Categorical(demo_data["id"])
    oh_data["id"] = pd.Categorical(oh_data["id"])
    
    demo_file = "nhanes_demo.pkl"
    dental_file = "nhanes_dental.pkl"
    
    dem_file = open(demo_file, "wb")
    pickle.dump(demo_data, dem_file)
    dem_file.close()
    
    dent_file = open(dental_file, "wb")
    pickle.dump(oh_data, dent_file)
    dent_file.close()
else:
    demo_data = pd.read_pickle(file_list[0])
    oh_data = pd.read_pickle(file_list[1])
# - 

# ## Part b: Merge the variable OHDDESTS into the demographics data
# We want to match all survey participants by their unique ids, but also want
# to ensure that they are matched up with data from the same cohort:

# +
oh_data["id"] = pd.Categorical(oh_data["id"])
demo_data["id"] = pd.Categorical(demo_data["id"])

dental_status = oh_data

demo_merged = pd.merge(demo_data, 
                       dental_status, 
                       on = ["id", "Cohort"],
                       how = "left")
# -

# Our next task is to create a clean dataset of the merged data with only
# select dimensions, and applying some logic to ensure that the amount of 
# "missingness" in the data is accurately represented.

# +
def college_status(u20, education):
    """

    Parameters
    ----------
    u20 : TYPE
        DESCRIPTION.
    education : TYPE
        DESCRIPTION.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    """
    out = 0
    if(u20):
        out = 0
    elif(education < 4):
        out = 0
    elif(education < 7):
        out = 1
    return out

def exam_stat(exam, ohx):
    """

    Parameters
    ----------
    exam : TYPE
        DESCRIPTION.
    ohx : TYPE
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    if(exam == 2 and ohx == 1):
        return 1
    else:
        return 0


demo_merged["under_20"] = demo_merged["age"].apply(lambda y: y < 20)

demo_merged["college"] = demo_merged.apply(lambda x: 
                                       college_status(x["under_20"], 
                                                      x["education"]),
                                                       axis = 1)

#dent_file = open("nhanes_clean_data.pkl", "wb")
#pickle.dump(demo_merged, dent_file)
#dent_file.close()
    
clean_df = demo_merged[["id", 
                        "gender", 
                        "age", 
                        "under_20", 
                        "college", 
                        "exam_status", 
                        "ohx_status"]]


# Make nicer labels
college_labels = {0: "No college/<20",
                  1: "Some college/college graduate"}

exam_labels = {0: "missing", 1: "complete"}
gender_labels ={1: "male", 2: "female"}
u20_labels ={True: "Under 20", False: "20+ yrs"}


clean_df.loc[:,"college"] = pd.Categorical(clean_df.loc[:,"college"]
                                           .replace(college_labels)
                                           .copy())

clean_df.loc[:,"ohx"] = clean_df.apply(lambda x: exam_stat(x["exam_status"], 
                                                      x["ohx_status"]),
                                                       axis = 1)

clean_df.loc[:,"ohx"] = pd.Categorical(clean_df["ohx"]
                                           .replace(exam_labels) # potentially move to line above
                                           .copy())

clean_df.loc[:, "gender"] = pd.Categorical(clean_df["gender"]
                                           .replace(gender_labels)
                                           .copy())

clean_df.loc[:, "under_20"] = pd.Categorical(clean_df["under_20"]
                                           .replace(u20_labels)
                                           .copy())
# -

# ## 1 C
# After cleaning the data, we can drop some of the values that do not 
# have completevalues in the "exam_status" variable.

# +
start_rows = len(clean_df.index)
clean_df = clean_df.drop(clean_df[clean_df["exam_status"] != 2].index)
end_rows = len(clean_df.index)

removal_str = ("After removing {0: } rows with missing exam status," + 
               "{1: } rows remain in the data.")

print(removal_str.format(start_rows - end_rows, end_rows))
# -

# ## 1 D
# In order to effectively find the count, mean, and standard deviation of the
# metrics requested, we can efficiently summarize the data using `pd.crosstab`.
# We will create some functions to help us:
    
# +
table_vals = ["under_20", "gender", "college", "age"]

def total_and_pct(x):
        """
        Parameters
        ----------
        x : DataFrame
            Data that will be summarized, finding the total for each row/column

        Returns
        -------
        pct : DataFrame
            Pandas data frame with the same dimensions as x; each element is 
            divided by its row/column total
        """
        total = x.sum()
        pct = x / total
        return pct
    
def ctab(factor, column, values = None, aggfunc = None, endfunc = None):
    """
    Convenient wrapper for pd.crosstab(), allowing for a transformation to
    be applied at the end
    
    Parameters
    ----------
    factor : pandas series
        maps to "index" argument of crosstab
    column : pandas series
        maps to "columns" argument of crosstab
    values : pandas series
        value field to summarize, required if aggfunc specified
    aggfunc : function
        function used to aggregate in place of "count" in crosstab
    endfunc : function
        Final function used to transform data after being passed to 
        the crosstab

    Returns
    -------
    DataFrame
        Crosstabulated summary of the data frame, with transformation by
        endfunc if specified

    """
    tab = pd.crosstab(factor, column, values, aggfunc=aggfunc)
    if endfunc:
        tab = tab.apply(endfunc) 
    return tab.applymap(lambda x: [x])
# -

# Next, we can create two separate tables, one for the count / percent metrics
# and another for the distribution metrics

# +    
count_tabs = {}

for val in table_vals[:-1]:
    count_tabs[val] = (ctab(clean_df[val], clean_df["ohx"]) +
                 ctab(clean_df[val], clean_df["ohx"], endfunc=total_and_pct))
    
dist_tabs = {}

dist_funcs =[np.mean, np.std]
try:
    clean_df["ohx"] = clean_df["ohx"].cat.add_categories("p_value")
except:
    pass

for val in table_vals[:-1]:
    dist_tabs[val] = (ctab(clean_df[val], 
                           clean_df["ohx"], 
                           values = clean_df["age"],
                           aggfunc = np.mean) +
                      ctab(clean_df[val],
                           clean_df["ohx"],
                           values = clean_df["age"],
                           aggfunc = np.std))

comb_count = pd.concat(count_tabs.values(), keys = count_tabs.keys())
comb_dist = pd.concat(dist_tabs.values(), keys = dist_tabs.keys())

comb_count.columns = comb_count.columns.add_categories("p_val")
comb_dist.columns = comb_dist.columns.add_categories("p_val")


p_vals = []
chi_sq = []

c_stats = []
m_stats = []

for row in range(len(comb_count)):
    c_stats = (comb_dist.iloc[row, 0].copy())
    c_stats.append(comb_dist.iloc[row, 1][0])
    
    m_stats = (comb_dist.iloc[row, 1].copy())
    m_stats.append(comb_count.iloc[row, 1][0])
    
    p_vals.append(stats.ttest_ind_from_stats(c_stats[0], 
                                             c_stats[1], 
                                             c_stats[2],
                                             m_stats[0], 
                                             m_stats[1], 
                                             m_stats[2],
                                             equal_var=False
                                             )[1]
                  )
    
 
    chi_sq.append(stats.chisquare((c_stats[2], m_stats[2]))[1])

count_table = comb_count.assign(p_val = chi_sq)
dist_table = comb_dist.assign(p_val = p_vals)

count_table["complete"] = (count_table["complete"]
                            .apply(lambda x: 
                            "{0:.0f}, ({1:0.2%})".format(*x)))
count_table["missing"] = (count_table["missing"]
                            .apply(lambda x: 
                            "{0:.0f}, ({1:0.2%})".format(*x)))
count_table["p_val"] = (count_table["p_val"]
                            .apply(lambda x: 
                            "{:.2e}".format(x)))
    
dist_table["complete"] = (dist_table["complete"]
                            .apply(lambda x: 
                            "{0:0.2f}, ({1:0.2f})".format(*x)))
dist_table["missing"] = (dist_table["missing"]
                            .apply(lambda x: 
                            "{0:.2f}, ({1:0.2f})".format(*x)))
dist_table["p_val"] = (dist_table["p_val"]
                            .apply(lambda x: 
                            "{:.2e}".format(x)))    

c_table = count_table.to_html()

d_table = dist_table.to_html()
# -

# Summary statistics (count, percent of total)
# with associated p-value for chi-squared tests

# +
display(HTML(c_table))
# -

# Distribution statistics (mean, std dev) with associated p-value for t tests

# +
display(HTML(d_table))
# -

# # Question 2: Monte Carlo Comparison


# +
ci_level = .95 
prop_vals = np.arange(0.05, 0.55, .05)
n_vals = [10, 30, 50, 100, 500, 1000, 4000]
mce = []

ci_methods = ["norm", "bin", "cp", "jeff", "ac"]
nice_labels = ["Normal",
               "Binomial",
              "Clopper-Pearson",
              "Jeffrey's",
              "Agresti-Coull"]

def num_mc_iters(lvl, p = 0.5, me = 0.005):
    """
    
    Parameters
    ----------
    lvl : TYPE
        DESCRIPTION.
    p : TYPE, optional
        DESCRIPTION. The default is 0.5.
    me : TYPE, optional
        DESCRIPTION. The default is 0.005.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    z = stats.norm.ppf((1-lvl)/2)
    return int(np.ceil(((z / me) ** 2) * p * (1-p)))


rng = np.random.default_rng(10 * 22 + 2021) 

interval_dict = {}

for x in n_vals:
    interval_dict[x] = {}
    for y in prop_vals:
        interval_dict[x][y] = {}
        iters = num_mc_iters(ci_level, 0.5)
        samples = []
        for i in range(1000):
            sample_set = rng.binomial(1, y, x)
            samples.append((x, y, iters, sample_set))
        for z in ci_methods:
            interval_dict[x][y][z] = {}
            for j, s in enumerate(samples):
                interval_dict[x][y][z][j] = ci.bin_est_interval(
                    s[3], 
                    cl = ci_level, 
                    method = z,
                    ci_format = None)


mc_results = pd.concat({
    key: pd.DataFrame.from_dict(val, "index") for key, val in interval_dict.items()
}, axis = 0)


mc_results.index.set_names(["N", "P"], inplace=True)
col_map = {"variable" : "method", "value" : "CI"}

unindexed = mc_results.reset_index().melt(id_vars = ["N", "P"]).rename(columns = col_map)

out = unindexed["CI"].to_dict()

intervals = pd.concat({
    key: pd.DataFrame.from_dict(val, "index") for key, val in out.items()
    }, axis = 0)


intervals = intervals.reset_index()
intervals = intervals.drop(["level_1"], axis = 1)
intervals = intervals.set_index("level_0")

final = pd.merge(intervals, 
                 unindexed, 
                 how = "inner",
                 left_index = True,
                 right_index = True)


final["width"] = final["upr"] - final["lwr"]


def in_bounds(lwr, upr, correct):
    """
    
    Parameters
    ----------
    lwr : TYPE
        DESCRIPTION.
    upr : TYPE
        DESCRIPTION.
    correct : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return lwr < correct < upr



final["in_bounds"] = final.apply(lambda x: in_bounds(x["lwr"], 
                                                     x["upr"], 
                                                     x["P"]),
                                 axis = 1                    
                                 )
# -


# Find actual confidence interval that each method achieves

# +
method_ints = final.groupby(["method", "N", "P", "in_bounds"]).agg("count")#.reset_index(1)

t = final.pivot_table(values = "est", index = ["method", "N", "P"], columns = "in_bounds", aggfunc = "count")

w = final.pivot_table(values = "width", index = ["method", "N", "P"], aggfunc = "mean")


t["Confidence Level"] = t.iloc[:, 1] / 1000

test = t.reset_index()



t = test.set_index("method")

def plot_contours_width(df, n, p, methods):
    fig, axs = plt.subplots(1,len(methods), figsize=(8,2))
    for i, m in enumerate(methods):
        print(m)
        width = df.loc[m].pivot_table(values = "width", 
                                   index = "P", 
                                   columns = "N")
        
        CS = axs[i].contour(n, p, width, label = m)
        axs.clabel(CS, inline = True, fontsize = 10)
        axs.autoscale()
    
    return fig, axs
        
def plot_contours_binned(df, n, p, methods):    
    fig, axs = plt.subplots(1,len(methods))
    for i, m in enumerate(methods):
        method_df = df.loc[m]
        
        
        bin_out = pd.cut(method_df["Confidence Level"], 5)
        
        bin_mid = [(a.left + a.right) / 2 for a in bin_out]
        
        method_df["c_level"] = bin_mid

        ci = method_df.pivot_table(values = "c_level", 
                                   index = "P", 
                                   columns = "N")

        CS = axs.contour(n, p, ci, label = m)
        axs.clabel(CS, inline = True, fontsize = 10)
        axs.autoscale()
    
    return fig, axs
# -

# + 

fig, ax = plot_contours_binned(t, n_vals, prop_vals, ["ac"])   

fig_width = plt.figure()
ax_1 = fig_width.add_subplot(111)

width = w.loc["norm"].pivot_table(values = "width", 
                                   index = "P", 
                                   columns = "N")
ax_1.contour(n_vals, prop_vals, width)
ax_1.set_xlabel("Normal")


ax_2 = fig_width.add_subplot(121)

width2 = w.loc["bin"].pivot_table(values = "width", 
                                   index = "P", 
                                   columns = "N")
ax_2.contour(n_vals, prop_vals, width2)
ax_2.set_xlabel("Binomial")

ax_3 = fig_width.add_subplot(131)

width3 = w.loc["jeff"].pivot_table(values = "width", 
                                   index = "P", 
                                   columns = "N")
ax_3.contour(n_vals, prop_vals, width3)
ax_3.set_xlabel("Jeffrey")


fig_width.show()
# -

plot_contours_binned(t, n_vals, prop_vals, ci_methods)

