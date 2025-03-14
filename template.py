import os

# Define the directory structure to create
directories = [
    "config",
    "data/input",
    "data/intermediate",
    "data/output",
    "data/graphs",
    "data/insights",
    "data/insights_other",
    "data/results",
    "src"
]

# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created or already exists: {directory}")

# Define the files to create (empty files or with minimal placeholders)
files = [
    "config/config.json",
    "src/etl.py",
    "src/analysis.py",
    "src/llm.py",
    "src/main.py",
    "src/insights_other.py",
    "src/file_processor.py",
    "requirements.txt"
]

# Create empty files (if they don't already exist)
for file in files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            # Optionally, write a simple placeholder text in each file
            f.write("# " + file + " - Add your code here\n")
        print(f"Created file: {file}")
    else:
        print(f"File already exists: {file}")

print("\nProject structure setup is complete!")
