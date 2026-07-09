import pandas as pd

# Load dataset
df = pd.read_csv("Sample - Superstore1.csv", encoding="latin1")

# View first 5 rows
print(df.head())

# Total Sales
total_sales = df["Sales"].sum()
print("Total Sales:", total_sales)

# Total Profit
total_profit = df["Profit"].sum()
print("Total Profit:", total_profit)

# Sales by Region
sales_by_region = df.groupby("Region")["Sales"].sum()
print("\nSales by Region")
print(sales_by_region)

# Profit by Region
profit_by_region = df.groupby("Region")["Profit"].sum()
print("\nProfit by Region")
print(profit_by_region)

# Sales by Category
sales_by_category = df.groupby("Category")["Sales"].sum()
print("\nSales by Category")
print(sales_by_category)

# Profit by Category
profit_by_category = df.groupby("Category")["Profit"].sum()
print("\nProfit by Category")
print(profit_by_category)

# Best Region & Category
best_region = sales_by_region.idxmax()
best_category = sales_by_category.idxmax()

print("\n----- BUSINESS INSIGHTS -----")
print(f"Total Sales: ${total_sales:,.2f}")
print(f"Total Profit: ${total_profit:,.2f}")
print(f"Best Region: {best_region}")
print(f"Best Category: {best_category}")


