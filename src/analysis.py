import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.feature_extraction.text import TfidfVectorizer

def find_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def load_json_data(directory):
    json_files = find_json_files(directory)
    data = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data.append(json.load(f))
    return data

def build_vector_store(categories):
    vectorizer = TfidfVectorizer(stop_words='english')
    category_embeddings = vectorizer.fit_transform(categories)
    return vectorizer, category_embeddings

def analyze_query_with_llm(user_query, categories, vectorizer, category_embeddings, generate_response_func):
    system_prompt = "You are an expert data analyst. Help refine this query to better match data categories."
    refined_query = generate_response_func(system_prompt, user_query)
    query_embedding = vectorizer.transform([refined_query]).toarray()
    similarities = np.dot(category_embeddings.toarray(), query_embedding.T).flatten()
    most_similar_idx = np.argmax(similarities)
    return categories[most_similar_idx]

def extract_values_for_category(json_data, category):
    values = []
    for file_data in json_data:
        if category in file_data:
            values.extend(file_data[category])
    return values

def plot_trend_and_save(values, category, output_file):
    # Extract dates and values from the list of dict entries
    dates = []
    y_values = []
    for entry in values:
        dates.append(entry["Date"])
        y_values.append(entry["value"])
    dates = pd.to_datetime(dates)
    df = pd.DataFrame({'ds': dates, 'y': y_values})
    
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=14, freq='M')
    forecast = model.predict(future)
    forecast_filtered = forecast[forecast['ds'] > max(dates)]
    
    plt.figure(figsize=(10,6))
    sns.lineplot(x=dates, y=y_values, label=f'{category} (Observed)', color='blue')
    sns.lineplot(x=forecast_filtered['ds'], y=forecast_filtered['yhat'],
                 label=f'{category} (Predicted)', color='orange', linestyle='--')
    if not forecast_filtered.empty:
        plt.plot([max(dates), forecast_filtered['ds'].iloc[0]],
                 [y_values[-1], forecast_filtered['yhat'].iloc[0]],
                 color='orange', linestyle='--')
    plt.title(f'Trend and Prediction for {category}')
    plt.xlabel('Time')
    plt.ylabel(category)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def display_predicted_values(values):
    dates = []
    y_values = []
    for entry in values:
        dates.append(entry["Date"])
        y_values.append(entry["value"])
    dates = pd.to_datetime(dates)
    df = pd.DataFrame({'ds': dates, 'y': y_values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=14, freq='M')
    forecast = model.predict(future)
    last_date = max(dates)
    forecast_filtered = forecast[forecast['ds'] > last_date]
    predicted_values = forecast_filtered[['ds', 'yhat']]
    print("Predicted Values for upcoming periods:")
    print(predicted_values.to_string(index=False))

def generate_insights_from_predictions(predicted_values, parameter, generate_response_func):
    summary_text = (
        f"The parameter '{parameter}' has been analyzed with the following predicted values "
        "for each month from the last observed date onward:\n\n"
    )
    summary_text += predicted_values.to_string(index=False, header=False)
    summary_text += (
        "\n\nProvide insights, recommendations, and suggestions to improve decision-making for this parameter."
    )
    system_prompt = (
        "You are a highly skilled business consultant specializing in data-driven decision-making. "
        "Analyze the provided predictions and generate actionable insights and recommendations."
    )
    insights = generate_response_func(system_prompt, summary_text, temperature=0.7, max_tokens=1000)
    print(f"Insights and Recommendations for '{parameter}':\n")
    print(insights)
    return insights


