"""
Rating Product & Sorting Reviews in Amazon

Business Problem
Trying to calculate product ratings more accurately and ordering product reviews more accurately

Dataset Story
This dataset, which includes Amazon product data, includes product categories and various metadata.
The product with the most comments in the Electronics category has user ratings and comments.

Variables
reviewerID – User ID
Ex: A2SUAM1J3GNN3B
asin – Product ID.
Ex: 0000013714
reviewerName – Username
helpful – Rating of helpful comments
Ex: 2/3
reviewText – Comment
User-written review text
overall – Product rating
summary – Review summary
unixReviewTime – Review time
Unix time
reviewTime – Review time
Raw
"""

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("amazon_review.csv")
df.head()

df["overall"].value_counts()
# Current average rating available
df["overall"].mean()
# 4.5875

df["reviewTime"].dtype
df['reviewTime'] = pd.to_datetime(df['reviewTime'])
df['reviewTime'].max()
current_date = pd.to_datetime('2014-12-08 0:0:0')
df["days_diff"] = (current_date - df['reviewTime']).dt.days
df.head()

df["reviewTime"].describe([0.25, 0.50, 0.75]).T
pd.cut(df["days_diff"], [25, 50, 75]).value_counts()

q25 = df["days_diff"].quantile(0.25)
q50 = df["days_diff"].quantile(0.50)
q75 = df["days_diff"].quantile(0.75)

# Let's focus on the one whose shopping date is closer.
average_rating = df.loc[df["days_diff"] <= 25, "overall"].mean() * 28 / 100 + \
df.loc[(df["days_diff"] > q25) & (df["days_diff"] <= q50), "overall"].mean() * 26 / 100 + \
df.loc[(df["days_diff"] > q50) & (df["days_diff"] <= q75), "overall"].mean() * 24 / 100 + \
df.loc[(df["days_diff"] > q75), "overall"].mean() * 22 / 100

average_rating
# 4.608649

# TASK 2
# Specify 20 reviews for the product to be displayed on the product detail page.
df.sort_values("helpful_yes", ascending=False).head(20)
df["up"] = df["helpful_yes"]
df["down"] = df["total_vote"] - df["helpful_yes"]

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["up"], x["down"]), axis = 1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score calculation process

     - The lower limit of the confidence interval to be calculated for the Bernoulli parameter P is accepted as the Wilson Lower Bound score.
     - The score to be calculated is used for product ranking.
     - If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and adjusted to Bernoulli.
     - This brings with it some problems. For this reason, it is necessary to use the bayesian average rating.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)
df.sort_values("wilson_lower_bound", ascending=False).head(20)
df.head(20)