import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

## 1. Data Wrangling

# Load the dataset
df = pd.read_csv("AusApparalSales4thQrt2020.csv")

# Check for missing values
print("Missing Values:")
print(df.isna().sum())

# Check data types and general structure
print("\nData Overview:")
print(df.info())

scaler = MinMaxScaler()
df['Sales_Normalized'] = scaler.fit_transform(df[['Sales']])

print("\nNormalized Sales Data:")
print(df[['Sales', 'Sales_Normalized']].head())

grouped_data = df.groupby(['Group', 'State']).agg({
    'Sales': ['sum', 'mean'],
    'Unit': ['sum', 'median']
}).reset_index()

print("\nGrouped Data Analysis:")
print(grouped_data.head())

## 2. Data Analysis

sales_stats = df['Sales'].describe()
units_stats = df['Unit'].describe()

print("Sales Statistics:")
print(sales_stats)

print("\nUnits Statistics:")
print(units_stats)

highest_sales_group = df.groupby('Group')['Sales'].sum().idxmax()
lowest_sales_group = df.groupby('Group')['Sales'].sum().idxmin()

print(f"\nHighest Sales Group: {highest_sales_group}")
print(f"Lowest Sales Group: {lowest_sales_group}")

# Convert date column to datetime (if not already)
df['Date'] = pd.to_datetime(df['Date'])

# Weekly Sales Report
weekly_sales = df.resample('W', on='Date')['Sales'].sum()

# Monthly Sales Report
monthly_sales = df.resample('ME', on='Date')['Sales'].sum()

print("\nWeekly Sales:")
print(weekly_sales)

print("\nMonthly Sales:")
print(monthly_sales)

## 3. Data Visualization

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='State', y='Sales', hue='Group', estimator=sum)
plt.title("State-wise Sales by Group")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Time', y='Sales', errorbar=None)
plt.title("Peak Sales Hours")
plt.show()

## 4. Report Generation

# Boxplot for Sales Distribution
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Group', y='Sales')
plt.title("Sales Distribution by Group")
plt.show()
