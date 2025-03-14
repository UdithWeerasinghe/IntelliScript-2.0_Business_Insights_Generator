import os
import json
from src import etl, analysis, llm

def main():
    # Define folder paths (adjust if needed)
    input_folder = "data/input"                # Place raw Excel files here
    intermediate_folder = "data/intermediate"    # Organized Excel files will be saved here
    output_folder = "data/output"                # JSON files and consolidated Excel will be saved here
    graphs_folder = "data/graphs"                # Graph images will be saved here
    insights_folder = "data/insights"            # Insights text files will be saved here

    # Ensure all folders exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(intermediate_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(graphs_folder, exist_ok=True)
    os.makedirs(insights_folder, exist_ok=True)

    # 1. Run the ETL pipeline
    #etl.run_etl_pipeline(input_folder, intermediate_folder, output_folder)

    # 2. Load JSON data for analysis from the output folder
    json_data = analysis.load_json_data(output_folder)
    categories = list({key for file_data in json_data for key in file_data.keys()})
    if not categories:
        raise ValueError("No valid categories found in the datasets.")
    print("Categories found:", categories)

    # 3. Build vector store for the categories
    vectorizer, category_embeddings = analysis.build_vector_store(categories)

    # 4. Authenticate and load the LLM model
    config = llm.load_config("config/config.json")
    model, tokenizer, device = llm.authenticate_and_load_model(config)
    generate_response_func = lambda sys_prompt, usr_prompt, temperature=0.7, max_tokens=500: llm.generate_response(
        sys_prompt, usr_prompt, temperature, max_tokens, model, tokenizer, device
    )

    # 5. Analyze the user query to identify the most relevant category
    user_query = "Garments"  # You may replace with input() for interactivity
    most_relevant_category = analysis.analyze_query_with_llm(user_query, categories, vectorizer, category_embeddings, generate_response_func)
    print("Most relevant category:", most_relevant_category)

    # 6. Extract values for the chosen category
    values = analysis.extract_values_for_category(json_data, most_relevant_category)

    # 7. Plot the trend with predictions and save the graph
    graph_file = os.path.join(graphs_folder, f"{most_relevant_category}_trend.png")
    analysis.plot_trend_and_save(values, most_relevant_category, graph_file)
    print(f"Graph saved at: {graph_file}")

    # 8. Display predicted values (optional)
    analysis.display_predicted_values(values)

    # 9. Prepare data for insights generation using Prophet
    import pandas as pd
    from prophet import Prophet
    dates = []
    y_values = []
    for entry in values:
        dates.append(entry["Date"])
        y_values.append(entry["value"])
    dates = pd.to_datetime(dates)
    df = pd.DataFrame({'ds': dates, 'y': y_values})
    prophet_model = Prophet()
    prophet_model.fit(df)
    future = prophet_model.make_future_dataframe(periods=14, freq='M')
    forecast = prophet_model.predict(future)
    forecast_filtered = forecast[forecast['ds'] > max(dates)]
    predicted_values_for_insights = forecast_filtered[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted Value'})

    # 10. Generate insights using the LLM and save them as a text file
    insights_text = analysis.generate_insights_from_predictions(predicted_values_for_insights, most_relevant_category, generate_response_func)
    insights_file = os.path.join(insights_folder, f"{most_relevant_category}_insights.txt")
    with open(insights_file, 'w') as f:
        f.write(insights_text)
    print(f"Insights saved at: {insights_file}")

     # ------------------ For Other File Types ------------------
    input_other_folder = "data/input_other"      # Raw files of other types
    other_text_folder = "data/other_text"          # Folder to save extracted texts
    other_results_folder = "data/other_results"    # Folder to save insights for other files

    os.makedirs(input_other_folder, exist_ok=True)
    os.makedirs(other_text_folder, exist_ok=True)
    os.makedirs(other_results_folder, exist_ok=True)

    from src import file_processor, insights_other

    # Process each file in input_other_folder: extract text and save it as a .txt file
    for file in os.listdir(input_other_folder):
        file_path = os.path.join(input_other_folder, file)
        text = file_processor.extract_text_from_file(file_path)
        if text:
            txt_filename = f"{os.path.splitext(file)[0]}.txt"
            txt_path = os.path.join(other_text_folder, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Extracted text saved to: {txt_path}")

    # Load the extracted texts and generate insights
    other_texts = insights_other.load_other_texts(other_text_folder)
    user_query_other = "How can we improve customer engagement?"  # Example query
    insights_other_text = insights_other.generate_other_insights(user_query_other, other_texts, generate_response_func)
    other_insights_file = os.path.join(other_results_folder, "other_insights.txt")
    with open(other_insights_file, "w", encoding="utf-8") as f:
        f.write(f"User Query: {user_query_other}\n\nGenerated Insights:\n{insights_other_text}")
    print(f"Other files insights saved at: {other_insights_file}")


if __name__ == "__main__":
    main()
