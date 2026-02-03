import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/Unemployment in India.csv")

# Display first rows
print("First 5 Rows")
print(df.head())

# Rename columns (remove spaces)
df.columns = df.columns.str.strip()

# Convert date column
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop missing values
df.dropna(inplace=True)

# Basic Info
print("\nDataset Info:")
print(df.info())

# ---------------------------
# Average Unemployment Rate by Region
# ---------------------------

region_avg = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean()

print("\nAverage Unemployment by Region:")
print(region_avg)

# Plot Region Wise Unemployment
plt.figure(figsize=(10,6))
region_avg.plot(kind='bar')
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/region_unemployment.png")
plt.show()

# ---------------------------
# Monthly Unemployment Trend
# ---------------------------

monthly_trend = df.groupby('Date')['Estimated Unemployment Rate (%)'].mean()

plt.figure(figsize=(12,6))
monthly_trend.plot()
plt.title("Monthly Unemployment Trend")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("output/monthly_trend.png")
plt.show()

# ---------------------------
# Heatmap for Correlation
# ---------------------------

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("output/correlation_heatmap.png")
plt.show()

print("\nAnalysis Completed Successfully!")
