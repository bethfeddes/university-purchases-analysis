# Student and Faculty Technology Purchase Pattern Analysis
Applies the Apriori algorithm to transactional computer retail data to discover association rules between university customer demographics and product purchases. Investigates the top ten rules for negative patterns.
# Dataset

# Requirements
- Python 3
- pandas
- mlxtend
# What it Does
- Bins customer age into decade ranges and one-hot encodes occupation
- Creates binary columns for each product in the Buys field
- Runs Apriori algorithm on nominalized data
- Sorts rules by lift and prints the top 10
- Applies the Kulczynski measure to each rule to identify any negative correlation patterns
# Key Findings
No negative patterns were found among the top ten rules in this dataset under the 0.5 threshold. The strongest rules in this dataset reflect positive correlations between the antecedents and the consequents.
# Configuration
Thresholds can be adjusted directly in the script:
- Minimum support: 0.1
- Minimum confidence: 0.8
- Epsilon (for Kulczynski): 0.5
