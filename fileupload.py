from google.colab import files

uploaded = files.upload()

# Get the name of the uploaded file
if uploaded:
    uploaded_file_name = list(uploaded.keys())[0]
    print(f"File '{uploaded_file_name}' uploaded successfully.")
else:
    print("No file was uploaded.")
