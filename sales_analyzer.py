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
print(df.notna().sum())

# Check data types and general structure
print("\nData Overview:")
print(df.info())

scaler = MinMaxScaler()
df['Normalized_Sales_Data'] = scaler.fit_transform(df[['Sales']])

print("\nNormalized Sales Data:")
print(df[['Sales', 'Normalized_Sales_Data']].head())

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

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Weekly Sales Report
weekly_sales = df.resample('W', on='Date')['Sales'].sum()
print("\nWeekly Sales:")
print(weekly_sales)

# Monthly Sales Report
monthly_sales = df.resample('ME', on='Date')['Sales'].sum()
print("\nMonthly Sales:")
print(monthly_sales)

# Convert 'Date' column to datetime format for time-based analysis
df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")

# Extract week, month, and quarter for grouping
df["Week"] = df["Date"].dt.isocalendar().week
df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter

## 3. Data Visualization

# State wise sales plotting
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='State', y='Sales', hue='Group', estimator=sum)
plt.title("State-wise Sales by Group")
plt.xticks(rotation=45)
plt.show()

# Peak Sales Hours plotting
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Time', y='Sales', errorbar=None)
plt.title("Peak Sales Hours")
plt.show()

## 4. Report Generation

# Create subplots for time-based sales analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Weekly Sales Plot
sns.lineplot(ax=axes[0,0], data=df, x='Week', y='Sales', hue='Group', estimator=sum)
axes[0,0].set_title("Weekly Sales Trend in Q4 2020")

# Monthly Sales Plot
sns.barplot(ax=axes[0,1], data=df, x='Month', y='Sales', hue='Group', estimator=sum)
axes[0,1].set_title("Monthly Sales in Q4 2020")

# Quarterly Sales Plot
sns.barplot(ax=axes[1,0], data=df, x='Quarter', y='Sales', hue='Group', estimator=sum)
axes[1,0].set_title("Quarterly Sales in Q4 2020")

# Boxplot for Sales Distribution
sns.boxplot(ax=axes[1,1], data=df, x='Group', y='Sales', hue='Group')
axes[1,1].set_title("Sales Distribution by Group")

plt.tight_layout()
plt.show()
