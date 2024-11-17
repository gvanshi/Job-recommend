import pandas as pd

# Load datasets
students_data = pd.read_excel('data/raw/students_data.xlsx')
educators_data = pd.read_excel('data/raw/educators_data.xlsx')
industry_data = pd.read_excel('data/raw/industry_data.xlsx')

# Clean data
students_data.dropna(inplace=True)
educators_data.dropna(inplace=True)
industry_data.dropna(inplace=True)

# Save cleaned data
students_data.to_csv('data/processed/cleaned_students_data.csv', index=False)
educators_data.to_csv('data/processed/cleaned_educators_data.csv', index=False)
industry_data.to_csv('data/processed/cleaned_industry_data.csv', index=False)

print("Data cleaning completed!")
