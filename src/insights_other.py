import os

def load_other_texts(folder_path: str) -> dict:
    """
    Load all text files from the specified folder and return a dictionary
    with keys as filenames and values as the file content.
    """
    texts = {}
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            path = os.path.join(folder_path, file)
            with open(path, "r", encoding="utf-8") as f:
                texts[file] = f.read()
    return texts

def generate_other_insights(user_query: str, texts: dict, generate_response_func) -> str:
    """
    Concatenate the texts (or otherwise process them) and generate insights based on a user query.
    The generate_response_func should be your LLM function that accepts a system prompt and a user prompt.
    """
    combined_text = "\n\n".join(texts.values())
    prompt = (
        f"Given the following text data:\n\n{combined_text}\n\n"
        f"User query: {user_query}\n\n"
        "Please provide actionable business insights based on the above data."
    )
    insights = generate_response_func("Generate business insights", prompt, temperature=0.7, max_tokens=1000)
    return insights

