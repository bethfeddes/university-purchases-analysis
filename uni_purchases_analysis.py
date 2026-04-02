

# Bethany Feddes
# Intro to Data Mining
# 03/27/2026


# Import and install packages
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load data set
data=pd.read_csv('computer_trans.csv')

# # Discretize 'Age' into bins 20-29, 30-39, 40-49, 50-59
age_bins = pd.cut(data['Age'], [20, 30, 40, 50, 60],
right=False)
age_cols = pd.crosstab(data.index, age_bins)


# # Create binary attributes for 'Buys'
data['Scanner'] = data['Buys'].apply(lambda x: 1 if
'Scanner' in x else 0)
data['Laptop'] = data['Buys'].apply(lambda x: 1 if 'Laptop'
in x else 0)
data['Printer'] = data['Buys'].apply(lambda x: 1 if
'Printer' in x else 0)
data['Desktop'] = data['Buys'].apply(lambda x: 1 if
'Desktop' in x else 0)
data['Mouse'] = data['Buys'].apply(lambda x: 1 if 'Mouse'
in x else 0)

# Cross-tabulation for 'Occupation' Column
oc_cols = pd.crosstab(data.index, data['Occupation'])

# Concatenate DataFrames
df = pd.concat([oc_cols, age_cols,
data[['Scanner','Laptop','Printer','Desktop','Mouse']]], axis=1)

# Display resulting DataFrame
print(df)

# Apply apriori algorithm to find frequent itemsets
# Minimum support = 0.1, confidence minimum threshold = 0.8.

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
rules_sorted = rules.sort_values(by="lift", ascending=False)

# Print top ten strong association rules according to lift
top_ten_rules = rules_sorted.head(10)

for index, row in top_ten_rules.iterrows():
    antecedent = ', '.join([str(x) for x in row['antecedents']])
    consequent = ', '.join([str(x) for x in row['consequents']])

    print(f"Rule: {{{antecedent}}} -> {{{consequent}}}")
    print(f"Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}\n")

""" IDENTIFY INTERDIMENSIONAL ASSOCIATION RULES
 Interdimensional association rules include multiple dimensions, but no repeated predicates. The interdimensional association rules for this data set are:  
  -Rule: {Student, Printer} -> {[20, 30)}  
  -Rule: {Laptop, Student} -> {[20, 30)}
"""

""" Identify the hybrid-dimensional association rules:
 Hybrid-dimensional association rules are multidimensional and allow for repeated predicates. The hybrid-dimensional rules for this data set are:  
  -Rule: {Scanner, [20, 30)} -> {Laptop, Printer}  
  -Rule: {Professor, Laptop, Printer} -> {Scanner}  
  -Rule: {Professor, Laptop, Scanner} -> {Printer}  
  -Rule: {Scanner, [20, 30)} -> {Printer}  
  -Rule: {Scanner, [20, 30), Laptop} -> {Printer}  
  -Rule: {Student, Printer} -> {Laptop, [20, 30)}  
  -Rule: {Laptop, Student} -> {[20, 30)}
"""

# Python Implementation of Kulcynski measure-based definition using 0.5 threshold for negative patterns

# Use Kulczynski's measure-based definition: (P(X|Y) + P(Y|X)) / 2
def kulcynski(support_x, support_y, support_xy):
    p_y_given_x = support_xy / support_x
    p_x_given_y = support_xy / support_y
    return (p_y_given_x + p_x_given_y) / 2

# Method to get support
def support(itemset, df):
    return df.apply(lambda row: itemset.issubset(set(row)), axis=1).mean()

epsilon = 0.5 # threshold

# Run Kulczynski on each rule in the top ten
for _, row in top_ten_rules.iterrows():

    # Get supports from DataFrame
    support_x = row["antecedent support"]
    support_y = row["consequent support"]
    support_xy = row["support"]

    # Plug in values for Kulcynski
    kulc = kulcynski(support_x, support_y, support_xy)

    # Display results
    print(f"Rule: {row['antecedents']} -> {row['consequents']}")
    print(f"Kulczynski: {kulc:.4f}")

    # If Kulczynski value less than epsilon, rule has negative correlation
    if kulc < epsilon:
        print(f"Negative pattern, {kulc} < {epsilon}\n")
    # If Kulczynski value greater than epsilon, rule has positive correlation
    else:
        print("Not a negative pattern\n")


# No found negative patterns in the top ten rules of this data set when epsilon is equal to 0.5
