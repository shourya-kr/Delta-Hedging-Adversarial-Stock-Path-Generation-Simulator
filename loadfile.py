import pandas as pd

# Use the name of the uploaded file to load the DataFrame
file_name = uploaded_file_name
df_feb05 = pd.read_csv(file_name)

print(f"Successfully loaded '{file_name}' into a DataFrame.")

# Display the first 5 rows to verify
display(df_feb05.head())
