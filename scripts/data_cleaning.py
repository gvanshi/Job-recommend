# Clean Students Data
students_data_cleaned = students_data.dropna()  # Drop rows with missing values
students_data_cleaned.columns = [col.strip() for col in students_data_cleaned.columns]  # Remove extra spaces
students_data_cleaned.to_csv('data/processed/cleaned_students_data.csv', index=False)

# Clean Educators Data
educators_data_cleaned = educators_data.dropna()
educators_data_cleaned.columns = [col.strip() for col in educators_data_cleaned.columns]
educators_data_cleaned.to_csv('data/processed/cleaned_educators_data.csv', index=False)

# Clean Industry Data
industry_data_cleaned = industry_data.dropna()
industry_data_cleaned.columns = [col.strip() for col in industry_data_cleaned.columns]
industry_data_cleaned.to_csv('data/processed/cleaned_industry_data.csv', index=False)

print("Data cleaning completed and saved in the 'processed' folder.")
