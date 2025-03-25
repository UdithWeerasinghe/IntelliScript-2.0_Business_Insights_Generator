# import os
# import json
# from src import etl, analysis, llm

# def main():
#     # Define folder paths (update as needed)
#     input_folder = "data/input"         # Raw Excel files
#     intermediate_folder = "data/intermediate"       # Organized Excel files will be saved here
#     output_folder = "data/output"             # JSON files and consolidated Excel will be saved here
#     graphs_folder = "data/graphs"         # Graph images will be saved here
#     insights_folder = "data/insights"     # Insights text files for Excel data

#     # Ensure folders exist
#     os.makedirs(input_folder, exist_ok=True)
#     os.makedirs(intermediate_folder, exist_ok=True)
#     os.makedirs(output_folder, exist_ok=True)
#     os.makedirs(graphs_folder, exist_ok=True)
#     os.makedirs(insights_folder, exist_ok=True)

#     # # 1. Run the ETL pipeline for Excel files
#     # #etl.run_etl_pipeline(input_folder, intermediate_folder, output_folder)

#     # # 2. Build FAISS index from JSON output in T3
#     # index, documents, embedder = analysis.build_faiss_index(output_folder)

#     # # 3. Set up and load the LLM model
#     # config = llm.load_config("config/config.json")
#     # model, tokenizer, device = llm.authenticate_and_load_model(config)
#     # # generate_response_func = lambda sys_prompt, usr_prompt, temperature=0.7, max_tokens=500: llm.generate_response(
#     # #     sys_prompt, usr_prompt, temperature, max_tokens, model, tokenizer, device
#     # # )
#     # generate_response_func = lambda sys_prompt, usr_prompt, temperature=0.7, max_tokens=500: llm.generate_response(
#     # sys_prompt, usr_prompt, temperature, max_tokens, model, tokenizer, device
#     # )


#     # # 4. Use FAISS-based query refinement to determine the best matching category
#     # user_query = "Agriculture exports"  # Example query; can replace with input()
#     # most_relevant_category = analysis.analyze_query_with_llm_faiss(user_query, generate_response_func, index, documents, embedder)
#     # print("Most relevant category:", most_relevant_category)

#     # # 5. Extract values for the chosen category from all JSON files in T3
#     # json_files = analysis.find_json_files(output_folder)
#     # def extract_values_for_category_from_files(files, category):
#     #     values = []
#     #     for file in files:
#     #         with open(file, 'r', encoding='utf-8') as f:
#     #             content = json.load(f)
#     #         data = content.get("data", {})
#     #         if category in data:
#     #             values.extend(data[category])
#     #     return values
#     # values = extract_values_for_category_from_files(json_files, most_relevant_category)

#     # 1. Run the ETL pipeline for Excel files
#     # etl.run_etl_pipeline(input_folder, intermediate_folder, output_folder)

#     # 2. Build FAISS index from JSON output in the output folder
#     index, documents, embedder = analysis.build_faiss_index(output_folder)

#     # 3. Set up the LLM "model" using Ollama (no actual model loaded)
#     config = llm.load_config("config/config.json")
#     # Our new Ollama setup does not load a model, so we ignore returned values:
#     _, _, _ = llm.authenticate_and_load_model(config)
    
#     # 4. Create a lambda for generating responses via Ollama
#     generate_response_func = lambda sys_prompt, usr_prompt, temperature=0.7, max_tokens=500: llm.generate_response(
#         sys_prompt, usr_prompt, temperature, max_tokens
#     )

#     # 5. Use FAISS-based query refinement to determine the best matching category
#     user_query = "Agriculture exports"  # Example query; can replace with input()
#     most_relevant_category = analysis.analyze_query_with_llm_faiss(user_query, generate_response_func, index, documents, embedder)
#     print("Most relevant category:", most_relevant_category)

#     # 6. Plot trend and save graph
#     graph_file = os.path.join(graphs_folder, f"{most_relevant_category}_trend.png")
#     analysis.plot_trend_and_save(values, most_relevant_category, graph_file)
#     print(f"Graph saved at: {graph_file}")

#     # 7. Display predicted values (optional)
#     analysis.display_predicted_values(values)

#     # 8. Prepare data for insights generation using Prophet
#     import pandas as pd
#     from prophet import Prophet
#     dates = []
#     y_values = []
#     for entry in values:
#         dates.append(entry["Date"])
#         y_values.append(entry["value"])
#     dates = pd.to_datetime(dates)
#     df = pd.DataFrame({'ds': dates, 'y': y_values})
#     prophet_model = Prophet()
#     prophet_model.fit(df)
#     future = prophet_model.make_future_dataframe(periods=14, freq='M')
#     forecast = prophet_model.predict(future)
#     forecast_filtered = forecast[forecast['ds'] > max(dates)]
#     predicted_values_for_insights = forecast_filtered[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted Value'})

#     # 9. Generate insights using the LLM and save them as a text file
#     insights_text = analysis.generate_insights_from_predictions(predicted_values_for_insights, most_relevant_category, generate_response_func)
#     insights_file = os.path.join(insights_folder, f"{most_relevant_category}_insights.txt")
#     with open(insights_file, "w", encoding="utf-8") as f:
#         f.write(insights_text)
#     print(f"Insights saved at: {insights_file}")

# if __name__ == "__main__":
#     main()

import os
import json
from src import etl, analysis, llm

def main():
    # Define folder paths (update as needed)
    input_folder = "data/input"         # Raw Excel files
    intermediate_folder = "data/intermediate"       # Organized Excel files will be saved here
    output_folder = "data/output"             # JSON files and consolidated Excel will be saved here
    graphs_folder = "data/graphs"         # Graph images will be saved here
    insights_folder = "data/insights"     # Insights text files for Excel data

    # Ensure folders exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(intermediate_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(graphs_folder, exist_ok=True)
    os.makedirs(insights_folder, exist_ok=True)

    # 1. Run the ETL pipeline for Excel files
    # etl.run_etl_pipeline(input_folder, intermediate_folder, output_folder)

    # 2. Build FAISS index from JSON output in the output folder
    index, documents, embedder = analysis.build_faiss_index(output_folder)

    # 3. Set up the LLM "model" using Ollama (no actual model loaded)
    config = llm.load_config("config/config.json")
    # Our new Ollama setup does not load a model, so we ignore returned values:
    _, _, _ = llm.authenticate_and_load_model(config)
    
    # 4. Create a lambda for generating responses via Ollama
    generate_response_func = lambda sys_prompt, usr_prompt, temperature=0.7, max_tokens=500: llm.generate_response(
        sys_prompt, usr_prompt, temperature, max_tokens
    )

    # 5. Use FAISS-based query refinement to determine the best matching category
    #user_query = "Agriculture exports"  # Example query; can replace with input()
    user_query = "Prosperity index in Sri Lanka 2020" 
    most_relevant_category = analysis.analyze_query_with_llm_faiss(user_query, generate_response_func, index, documents, embedder)
    print("Most relevant category:", most_relevant_category)

    # 6. Extract values for the chosen category from all JSON files in the output folder
    json_files = analysis.find_json_files(output_folder)
    def extract_values_for_category_from_files(files, category):
        values = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            data = content.get("data", {})
            if category in data:
                values.extend(data[category])
        return values
    values = extract_values_for_category_from_files(json_files, most_relevant_category)  # <-- New step

    # 7. Plot trend and save graph
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
    with open(insights_file, "w", encoding="utf-8") as f:
        f.write(insights_text)
    print(f"Insights saved at: {insights_file}")

if __name__ == "__main__":
    main()
