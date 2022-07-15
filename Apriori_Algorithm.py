#Muhammad Hamza

from mlxtend.frequent_patterns import association_rules, apriori
import pandas as pd


def apriori_algorithm():
    df = pd.read_csv("chaldalfruits.csv")

    transactions_str = df.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Count')
    my_basket = transactions_str.pivot_table(index='Transaction', columns='Item', values='Count',aggfunc='sum').fillna(0)

    def encode(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    my_basket_sets = my_basket.applymap(encode)

    frequent_items = apriori(my_basket_sets, min_support=0.02, use_colnames=True)

    rules = association_rules(frequent_items, metric="lift", min_threshold=1)

    rules.sort_values('confidence', ascending=False, inplace=True)

    rules.sort_values('confidence', ascending=False)

    return frequent_items, rules

print(apriori_algorithm())