"""A simple data analysis pipeline. Run with: python analysis.py"""

from pathlib import Path
import pandas as pd

# 1. LOAD: Read the CSV file into a table called data.
# Path finds the CSV in the same folder as this Python file.
data_path = Path(__file__).resolve().parent / "insurance.csv"
data = pd.read_csv(data_path)

# 2. EXPLORE: See what the table contains.
print("\nFirst five rows:")
print(data.head())
print("\nNumber of rows and columns:")
print(data.shape)

# 3. CHECK: Look for missing values and repeated rows.
print("\nMissing values in each column:")
print(data.isna().sum())
print("\nNumber of duplicate rows:")
print(data.duplicated().sum())
# Keep the original records. A repeated row is not always a data error.

# 4. SUMMARIZE: Calculate basic statistics for charges.
average_charge = data["charges"].mean()
median_charge = data["charges"].median()
maximum_charge = data["charges"].max()

print("\nAverage charge:", round(average_charge, 2))
print("Median charge:", round(median_charge, 2))
print("Highest charge:", round(maximum_charge, 2))

# 5. COMPARE: Calculate average charges for each group.
# groupby("smoker") separates yes and no rows.
# ["charges"] selects the column. mean() calculates each group's average.
smoking_averages = data.groupby("smoker")["charges"].mean()
print("\nAverage charges by smoking status:")
print(smoking_averages.round(2))

region_averages = data.groupby("region")["charges"].mean()
print("\nAverage charges by region:")
print(region_averages.round(2))

# 6. INTERPRET: Use the calculated numbers to describe a finding.
smoker_average = smoking_averages["yes"]
nonsmoker_average = smoking_averages["no"]
charge_ratio = smoker_average / nonsmoker_average

print("\nFinding:")
print("Smokers' average charges are", round(charge_ratio, 2),
      "times non-smokers' average charges in this dataset.")
print("This comparison shows an association; it does not prove causation.")
