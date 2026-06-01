import kagglehub

# Download latest version
path = kagglehub.dataset_download("muratkokludataset/dry-bean-dataset")

print("Path to dataset files:", path)