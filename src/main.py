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

######################################################################################################################################################

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
#     #user_query = "Agriculture exports"  # Example query; can replace with input()
#     user_query = "Prosperity index in Sri Lanka 2020" 
#     most_relevant_category = analysis.analyze_query_with_llm_faiss(user_query, generate_response_func, index, documents, embedder)
#     print("Most relevant category:", most_relevant_category)

#     # 6. Extract values for the chosen category from all JSON files in the output folder
#     json_files = analysis.find_json_files(output_folder)
#     def extract_values_for_category_from_files(files, category):
#         values = []
#         for file in files:
#             with open(file, 'r', encoding='utf-8') as f:
#                 content = json.load(f)
#             data = content.get("data", {})
#             if category in data:
#                 values.extend(data[category])
#         return values
#     values = extract_values_for_category_from_files(json_files, most_relevant_category)  # <-- New step

#     # 7. Plot trend and save graph
#     graph_file = os.path.join(graphs_folder, f"{most_relevant_category}_trend.png")
#     analysis.plot_trend_and_save(values, most_relevant_category, graph_file)
#     print(f"Graph saved at: {graph_file}")

#     # 8. Display predicted values (optional)
#     analysis.display_predicted_values(values)

#     # 9. Prepare data for insights generation using Prophet
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

#     # 10. Generate insights using the LLM and save them as a text file
#     insights_text = analysis.generate_insights_from_predictions(predicted_values_for_insights, most_relevant_category, generate_response_func)
#     insights_file = os.path.join(insights_folder, f"{most_relevant_category}_insights.txt")
#     with open(insights_file, "w", encoding="utf-8") as f:
#         f.write(insights_text)
#     print(f"Insights saved at: {insights_file}")

# if __name__ == "__main__":
#     main()

########ollama code###################################################################################################################################################

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

#     # 1. Run the ETL pipeline for Excel files
#     # etl.run_etl_pipeline(input_folder, intermediate_folder, output_folder)

#     # 2. Build FAISS index from JSON output in the output folder
#     index, documents, embedder = analysis.build_faiss_index(output_folder)

#     # 3. Set up the LLM "model" using Ollama (no actual model loaded)
#     config = llm.load_config("config/config.json")
#     # For Ollama, we ignore model loading.
#     _, _, _ = llm.authenticate_and_load_model(config)
    
#     # 4. Create a lambda for generating responses via Ollama
#     generate_response_func = lambda sys_prompt, usr_prompt, temperature=0.7, max_tokens=500: llm.generate_response(
#         sys_prompt, usr_prompt, temperature, max_tokens
#     )

#     # 5. Use FAISS-based query refinement to determine the best matching category
#     user_query = "Agriculutural Exports" 


#     most_relevant_category = analysis.analyze_query_with_llm_faiss(
#         user_query, generate_response_func, index, documents, embedder
#     )
#     print("Most relevant category:", most_relevant_category)

#     # 6. Extract values for the chosen category from all JSON files in the output folder
#     json_files = analysis.find_json_files(output_folder)
#     def extract_values_for_category_from_files(files, category):
#         values = []
#         for file in files:
#             with open(file, 'r', encoding='utf-8') as f:
#                 content = json.load(f)
#             data = content.get("data", {})
#             if category in data:
#                 values.extend(data[category])
#         return values
#     excel_values = extract_values_for_category_from_files(json_files, most_relevant_category)

#     # Check if we have enough Excel data (at least 2 valid rows) for time series analysis.
#     if len(excel_values) < 2:
#         print("Not enough Excel data for time series analysis. Switching to non-excel insights.")

#         # Define folders for other file types
#         input_other_folder = "data/input_other"      # Raw files of other types
#         other_text_folder = "data/other_text"          # Folder to save extracted texts
#         other_results_folder = "data/other_results"    # Folder to save insights for other files

#         os.makedirs(input_other_folder, exist_ok=True)
#         os.makedirs(other_text_folder, exist_ok=True)
#         os.makedirs(other_results_folder, exist_ok=True)

#         from src import file_processor, insights_other

#         # Process each file in input_other_folder: extract text and save as a .txt file
#         for file in os.listdir(input_other_folder):
#             file_path = os.path.join(input_other_folder, file)
#             text = file_processor.extract_text_from_file(file_path)
#             if text:
#                 txt_filename = f"{os.path.splitext(file)[0]}.txt"
#                 txt_path = os.path.join(other_text_folder, txt_filename)
#                 with open(txt_path, "w", encoding="utf-8") as f:
#                     f.write(text)
#                 print(f"Extracted text saved to: {txt_path}")

#         # Load the extracted texts and generate insights for non-excel files
#         from src import insights_other
#         other_texts = insights_other.load_other_texts(other_text_folder)
#         # Use the same user query (or change as needed)
#         user_query_other = user_query
#         insights_other_text = insights_other.generate_other_insights(user_query_other, other_texts, generate_response_func)
#         other_insights_file = os.path.join(other_results_folder, "other_insights.txt")
#         with open(other_insights_file, "w", encoding="utf-8") as f:
#             f.write(f"User Query: {user_query_other}\n\nGenerated Insights:\n{insights_other_text}")
#         print(f"Other files insights saved at: {other_insights_file}")
#     else:
#         # 7. Plot trend and save graph (Excel branch)
#         graph_file = os.path.join(graphs_folder, f"{most_relevant_category}_trend.png")
#         analysis.plot_trend_and_save(excel_values, most_relevant_category, graph_file)
#         print(f"Graph saved at: {graph_file}")

#         # 8. Display predicted values (optional)
#         analysis.display_predicted_values(excel_values)

#         # 9. Prepare data for insights generation using Prophet
#         import pandas as pd
#         from prophet import Prophet
#         dates = []
#         y_values = []
#         for entry in excel_values:
#             dates.append(entry["Date"])
#             y_values.append(entry["value"])
#         dates = pd.to_datetime(dates)
#         df = pd.DataFrame({'ds': dates, 'y': y_values})
#         prophet_model = Prophet()
#         prophet_model.fit(df)
#         future = prophet_model.make_future_dataframe(periods=14, freq='M')
#         forecast = prophet_model.predict(future)
#         forecast_filtered = forecast[forecast['ds'] > max(dates)]
#         predicted_values_for_insights = forecast_filtered[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted Value'})

#         # 10. Generate insights using the LLM and save them as a text file (Excel branch)
#         insights_text = analysis.generate_insights_from_predictions(predicted_values_for_insights, most_relevant_category, generate_response_func)
#         insights_file = os.path.join(insights_folder, f"{most_relevant_category}_insights.txt")
#         with open(insights_file, "w", encoding="utf-8") as f:
#             f.write(insights_text)
#         print(f"Insights saved at: {insights_file}")

# if __name__ == "__main__":
#     main()


################3multi modal code - work fine#####################

# import os
# import json
# from src import etl, analysis, llm
# import datetime

# def main():
#     # Define folder paths
#     input_folder = "data/input"             # Raw Excel files
#     intermediate_folder = "data/intermediate"  # Organized Excel files
#     output_folder = "data/output"           # JSON files and consolidated Excel
#     graphs_folder = "data/graphs"           # Graph images for Excel data
#     insights_folder = "data/insights"       # Insights text files for Excel data
    
#     input_other_folder = "data/input_other"   # Raw files for other types (pdf, docx, etc.)
#     other_text_folder = "data/other_text"       # Extracted text files from other file types
#     other_results_folder = "data/other_results" # Generated insights for other file types

#     # Ensure all folders exist
#     for folder in [input_folder, intermediate_folder, output_folder, graphs_folder, insights_folder,
#                    input_other_folder, other_text_folder, other_results_folder]:
#         os.makedirs(folder, exist_ok=True)

#     # 1. Run the ETL pipeline for Excel files
#     #etl.run_etl_pipeline(input_folder, intermediate_folder, output_folder)

#     # 2. Build FAISS index from Excel JSON output
#     index, documents, embedder = analysis.build_faiss_index(output_folder)

#     # 3. Set up LLM using Ollama (dummy model load)
#     config = llm.load_config("config/config.json")
#     _, _, _ = llm.authenticate_and_load_model(config)
#     generate_response_func = lambda sys_prompt, usr_prompt, temperature=0.7, max_tokens=500: llm.generate_response(
#         sys_prompt, usr_prompt, temperature, max_tokens
#     )

#     # 4. Determine best matching category from Excel data using FAISS
#     user_query = "seafood exports"  # Example query; replace as needed
#     # user_query = "Exchange Rates for USD,GBP and EUR"  # Example query; replace as needed
#     #user_query = "Earnings from tourism"  # Example query; replace as needed
#     most_relevant_category = analysis.analyze_query_with_llm_faiss(user_query, generate_response_func, index, documents, embedder)
#     print("Most relevant category from Excel:", most_relevant_category)

#     # 5. Extract Excel values for the chosen category
#     json_files = analysis.find_json_files(output_folder)
#     def extract_values_for_category_from_files(files, category):
#         values = []
#         for file in files:
#             with open(file, 'r', encoding='utf-8') as f:
#                 content = json.load(f)
#             data = content.get("data", {})
#             if category in data:
#                 values.extend(data[category])
#         return values
#     excel_values = extract_values_for_category_from_files(json_files, most_relevant_category)

#     # 6. Check if Excel data is sufficient (at least 2 rows)
#     if len(excel_values) < 2:
#         print("Insufficient Excel data. Processing other file types for insights only.")
#         # Process other file types
#         from src import file_processor, insights_other
#         # For each file in input_other_folder, extract text and save as .txt
#         for file in os.listdir(input_other_folder):
#             file_path = os.path.join(input_other_folder, file)
#             text = file_processor.extract_text_from_file(file_path)
#             if text:
#                 txt_filename = f"{os.path.splitext(file)[0]}.txt"
#                 txt_path = os.path.join(other_text_folder, txt_filename)
#                 with open(txt_path, "w", encoding="utf-8") as f:
#                     f.write(text)
#                 print(f"Extracted text saved to: {txt_path}")
#         # other_texts = insights_other.load_other_texts(other_text_folder)
#         # insights_other_text = insights_other.generate_other_insights(user_query, other_texts, generate_response_func)
#         # other_insights_file = os.path.join(other_results_folder, "other_insights.txt")
#         # with open(other_insights_file, "w", encoding="utf-8") as f:
#         #     f.write(f"User Query: {user_query}\n\nGenerated Insights:\n{insights_other_text}")
#         # print(f"Insights for other files saved at: {other_insights_file}")

#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         other_insights_file = os.path.join(other_results_folder, f"other_insights_{timestamp}.txt")
#         with open(other_insights_file, "w", encoding="utf-8") as f:
#             f.write(f"User Query: {user_query}\n\nGenerated Insights:\n{insights_other}")
#         print(f"Other files insights saved at: {other_insights_file}")
#     else:
#         # 7. Plot trend and save graph for Excel data
#         graph_file = os.path.join(graphs_folder, f"{most_relevant_category}_trend.png")
#         analysis.plot_trend_and_save(excel_values, most_relevant_category, graph_file)
#         print(f"Graph saved at: {graph_file}")
#         # 8. Display predicted values (optional)
#         analysis.display_predicted_values(excel_values)
#         # 9. Generate insights for Excel data using Prophet
#         import pandas as pd
#         from prophet import Prophet
#         dates = []
#         y_values = []
#         for entry in excel_values:
#             dates.append(entry["Date"])
#             y_values.append(entry["value"])
#         dates = pd.to_datetime(dates)
#         df = pd.DataFrame({'ds': dates, 'y': y_values})
#         prophet_model = Prophet()
#         prophet_model.fit(df)
#         future = prophet_model.make_future_dataframe(periods=14, freq='M')
#         forecast = prophet_model.predict(future)
#         forecast_filtered = forecast[forecast['ds'] > max(dates)]
#         predicted_values_for_insights = forecast_filtered[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted Value'})
#         insights_text = analysis.generate_insights_from_predictions(predicted_values_for_insights, most_relevant_category, generate_response_func)
#         insights_file = os.path.join(insights_folder, f"{most_relevant_category}_insights.txt")
#         with open(insights_file, "w", encoding="utf-8") as f:
#             f.write(insights_text)
#         print(f"Insights saved at: {insights_file}")

# if __name__ == "__main__":
#     main()


####################################mm2##########################################################

# import os, json, datetime
# import pandas as pd
# from prophet import Prophet
# from src import etl, analysis, llm, file_processor, insights_other

# def main():
#     # 1) Paths
#     excel_in  = "data/input"
#     excel_mid = "data/intermediate"
#     excel_out = "data/output"
#     other_in  = "data/input_other"
#     other_txt = "data/other_text"
#     other_res = "data/other_results"
#     graphs    = "data/graphs"
#     insights  = "data/insights"

#     for p in [excel_in, excel_mid, excel_out, other_in, other_txt, other_res, graphs, insights]:
#         os.makedirs(p, exist_ok=True)

#     # 2) ETL Excel → JSON
#     #etl.run_etl_pipeline(excel_in, excel_mid, excel_out)

#     # 3) Extract text from other files
#     for fname in os.listdir(other_in):
#         path = os.path.join(other_in, fname)
#         text = file_processor.extract_text_from_file(path)
#         if text:
#             txt = os.path.splitext(fname)[0] + ".txt"
#             open(os.path.join(other_txt, txt), "w", encoding="utf-8").write(text)

#     # 4) Build unified FAISS index
#     index, docs, embedder = analysis.build_unified_index(excel_out, other_txt)

#     # 5) LLM setup (Ollama)
#     config = llm.load_config("config/config.json")
#     _, _, _ = llm.authenticate_and_load_model(config)
#     gen = lambda sys, usr: llm.generate_response(sys, usr)

#     # 6) Query & retrieve
#     user_query = input("Enter your query: ")
#     retrieved = analysis.retrieve_relevant_docs(user_query, index, docs, embedder, top_k=5)

#     # # 7) Separate time-series vs text
#     # ts_docs   = [d for d in retrieved if d["type"] == "timeseries"]
#     # text_docs = [d for d in retrieved if d["type"] == "text"]

#     # # Always record sources
#     # sources = set()
#     # for d in retrieved:
#     #     sources.add(d["file"] + (f"/{d['sheet']}" if d["type"]=="timeseries" else ""))

#     # # 8a) If any time-series doc found → forecasting + graph + insights
#     # if ts_docs:
#     #     best = ts_docs[0]  # top match
#     #     # Build DataFrame
#     #     df_ts = pd.DataFrame(best["series"])
#     #     df_ts['Date'] = pd.to_datetime(df_ts['Date'])
#     #     df_ts = df_ts.sort_values('Date')

#     #     # Rename columns in place for Prophet
#     #     df_prophet = df_ts.rename(columns={'Date': 'ds', 'value': 'y'})

#     #     # Fit Prophet
#     #     prophet_model = Prophet()
#     #     prophet_model.fit(df_prophet)

#     #     # Make future dataframe and predict
#     #     future = prophet_model.make_future_dataframe(periods=14, freq='M')
#     #     fc = prophet_model.predict(future)

#     #     # Filter only the forecast beyond the last observed date
#     #     last_ds = df_prophet['ds'].max()
#     #     fc_filt = fc[fc['ds'] > last_ds]

#     #     # Plot and save
#     #     import matplotlib.pyplot as plt
#     #     plt.figure(figsize=(8, 4))
#     #     plt.plot(df_prophet['ds'], df_prophet['y'], label='Observed')
#     #     plt.plot(fc_filt['ds'], fc_filt['yhat'], '--', label='Forecast')
#     #     plt.legend()
#     #     graph_file = os.path.join(graphs, f"{best['category']}_trend.png")
#     #     plt.savefig(graph_file)
#     #     plt.close()

#     #     # Insights prompt
#     #     summary = (
#     #         f"Category '{best['category']}' from {best['file']}/{best['sheet']}:\n"
#     #         f"Last observed: {df_prophet['y'].iloc[-1]} on {df_prophet['ds'].iloc[-1].date()}\n"
#     #         "Forecast:\n" +  
#     #         "\n".join(f"{row['ds'].date()}: {row['yhat']:.2f}" for _, row in fc_filt.iterrows())
#     #     )
#     #     sys_prompt = "You are a business consultant."
#     #     user_prompt = summary + "\n\nProvide actionable insights and recommendations."
#     #     insight_text = gen(sys_prompt, user_prompt)

#     #     # Save graph + insights
#     #     print("Graph saved:", graph_file)
#     #     ins_path = os.path.join(insights, f"{best['category']}_insights.txt")
#     #     with open(ins_path, "w", encoding="utf-8") as f:
#     #         f.write("Sources:\n" + "\n".join(sources) + "\n\n")
#     #         f.write(insight_text)
#     #     print("Insights saved:", ins_path)

#     # # 8b) Otherwise → text‑only insights
#     # else:
#     #     combined = "\n\n".join(d["text"] for d in text_docs)
#     #     sys_prompt = "You are a business consultant."
#     #     user_prompt = (
#     #         f"Sources: {', '.join(sources)}\n\n"
#     #         f"Relevant content:\n{combined}\n\n"
#     #         f"User query: {user_query}\n\n"
#     #         "Provide actionable insights and recommendations."
#     #     )
#     #     insight_text = gen(sys_prompt, user_prompt)

#     #     # Unique filename
#     #     ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     #     out = os.path.join(other_res, f"other_insights_{ts}.txt")
#     #     with open(out, "w", encoding="utf-8") as f:
#     #         f.write(insight_text)
#     #     print("Text-only insights saved:", out)

#         # 6) Separate time‑series vs text, but only keep those with ≥2 points
#     ts_docs   = [
#         d for d in retrieved
#         if d["type"] == "timeseries" and isinstance(d.get("series"), list) and len(d["series"]) >= 2
#     ]
#     text_docs = [d for d in retrieved if d["type"] == "text"]

#     # 7a) If we have at least one valid time‑series doc → forecasting + graph + insights
#     if ts_docs:
#         best = ts_docs[0]
#         # Build DataFrame safely
#         df_ts = pd.DataFrame(best["series"])
#         # Now 'Date' must exist
#         df_ts['Date'] = pd.to_datetime(df_ts['Date'])
#         df_ts = df_ts.sort_values('Date')

#         # Rename for Prophet
#         df_prophet = df_ts.rename(columns={'Date':'ds','value':'y'})

#         # Forecast
#         prophet_model = Prophet()
#         prophet_model.fit(df_prophet)
#         future = prophet_model.make_future_dataframe(periods=14, freq='M')
#         fc = prophet_model.predict(future)
#         last_ds = df_prophet['ds'].max()
#         fc_filt = fc[fc['ds'] > last_ds]

#         # Plot
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(8,4))
#         plt.plot(df_prophet['ds'], df_prophet['y'], label='Observed')
#         plt.plot(fc_filt['ds'], fc_filt['yhat'], '--', label='Forecast')
#         plt.legend()
#         graph_file = os.path.join(graphs, f"{best['category']}_trend.png")
#         plt.savefig(graph_file)
#         plt.close()
#         print("Graph saved:", graph_file)

#         # Prepare summary & insights
#         summary = (
#             f"Category '{best['category']}' from {best['file']}/{best['sheet']}:\n"
#             f"Last observed: {df_prophet['y'].iloc[-1]} on {df_prophet['ds'].iloc[-1].date()}\n"
#             "Forecast:\n" +
#             "\n".join(f"{row['ds'].date()}: {row['yhat']:.2f}"
#                       for _, row in fc_filt.iterrows())
#         )
#         insight_text = gen("You are a business consultant.", summary + "\n\nProvide actionable insights.")

#         # Save insights
#         ins_path = os.path.join(insights, f"{best['category']}_insights.txt")
#         with open(ins_path, "w", encoding="utf-8") as f:
#             f.write("Sources:\n" + "\n".join(sources) + "\n\n")
#             f.write(insight_text)
#         print("Insights saved:", ins_path)

#     # 7b) Otherwise → text‑only insights
#     else:
#         combined = "\n\n".join(d["text"] for d in text_docs)
#         prompt = (
#             f"Sources: {', '.join(sources)}\n\n"
#             f"Relevant content:\n{combined}\n\n"
#             f"User query: {user_query}\n\n"
#             "Provide actionable insights and recommendations."
#         )
#         insight_text = gen("You are a business consultant.", prompt)

#         ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         out = os.path.join(other_res, f"other_insights_{ts}.txt")
#         with open(out, "w", encoding="utf-8") as f:
#             f.write(insight_text)
#         print("Text-only insights saved:", out)


# if __name__=="__main__":
#     main()

###################################################################
# import os
# import json
# import datetime
# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt

# from src import etl, analysis, llm, file_processor

# def main():
#     # 1) Paths
#     excel_in  = "data/input"
#     excel_mid = "data/intermediate"
#     excel_out = "data/output"
#     other_in  = "data/input_other"
#     other_txt = "data/other_text"
#     other_res = "data/other_results"
#     graphs    = "data/graphs"
#     insights  = "data/insights"

#     # Ensure all folders exist
#     for p in [excel_in, excel_mid, excel_out, other_in, other_txt, other_res, graphs, insights]:
#         os.makedirs(p, exist_ok=True)

#     # 2) ETL Excel → JSON (uncomment if you want to run)
#     # etl.run_etl_pipeline(excel_in, excel_mid, excel_out)

#     # 3) Extract text from other files
#     for fname in os.listdir(other_in):
#         path = os.path.join(other_in, fname)
#         text = file_processor.extract_text_from_file(path)
#         if text:
#             txt = os.path.splitext(fname)[0] + ".txt"
#             with open(os.path.join(other_txt, txt), "w", encoding="utf-8") as f:
#                 f.write(text)

#     # 4) Build unified FAISS index
#     index, docs, embedder = analysis.build_unified_index(excel_out, other_txt)

#     # 5) LLM setup (Ollama)
#     config = llm.load_config("config/config.json")
#     _, _, _ = llm.authenticate_and_load_model(config)
#     gen = lambda sys, usr: llm.generate_response(sys, usr)

#     # 6) Query & retrieve
#     user_query = input("Enter your query: ")
#     retrieved = analysis.retrieve_relevant_docs(user_query, index, docs, embedder, top_k=5)

#     # Build 'sources' list
#     sources = []
#     for d in retrieved:
#         if d["type"] == "timeseries":
#             sources.append(f"{d['file']}/{d['sheet']}")
#         else:
#             sources.append(d["file"])
#     seen = set()
#     sources = [s for s in sources if not (s in seen or seen.add(s))]

#     # Separate time‑series vs text, requiring ≥2 points
#     ts_docs   = [
#         d for d in retrieved
#         if d["type"] == "timeseries"
#            and isinstance(d.get("series"), list)
#            and len(d["series"]) >= 2
#     ]
#     text_docs = [d for d in retrieved if d["type"] == "text"]

#     # 7a) If we have at least one valid time‑series doc → forecasting + graph + insights
#     if ts_docs:
#         best = ts_docs[0]

#         # Build DataFrame from the series
#         df_ts = pd.DataFrame(best["series"])
#         df_ts['Date'] = pd.to_datetime(df_ts['Date'])
#         df_ts = df_ts.sort_values('Date')

#         # Rename for Prophet
#         df_prophet = df_ts.rename(columns={'Date':'ds','value':'y'})

#         # Forecast
#         prophet_model = Prophet()
#         prophet_model.fit(df_prophet)
#         future = prophet_model.make_future_dataframe(periods=14, freq='M')
#         fc = prophet_model.predict(future)
#         last_ds = df_prophet['ds'].max()
#         fc_filt = fc[fc['ds'] > last_ds]

#         # Plot with connector line
#         plt.figure(figsize=(10, 6))
#         # Observed
#         plt.plot(
#             df_prophet['ds'],
#             df_prophet['y'],
#             label=f'{best["category"]} (Observed)',
#             linewidth=2
#         )
#         # Forecast
#         plt.plot(
#             fc_filt['ds'],
#             fc_filt['yhat'],
#             linestyle='--',
#             label=f'{best["category"]} (Forecast)',
#             linewidth=2
#         )
#         # Connector
#         if not fc_filt.empty:
#             plt.plot(
#                 [df_prophet['ds'].iloc[-1], fc_filt['ds'].iloc[0]],
#                 [df_prophet['y'].iloc[-1], fc_filt['yhat'].iloc[0]],
#                 linestyle='--'
#             )
#         plt.title(f'Trend & Forecast for {best["category"]}')
#         plt.xlabel('Date')
#         plt.ylabel(best["category"])
#         plt.legend()
#         plt.tight_layout()
#         graph_file = os.path.join(graphs, f"{best['category']}_trend.png")
#         plt.savefig(graph_file)
#         plt.close()
#         print("Graph saved:", graph_file)

#         # Prepare summary & insights
#         summary = (
#             f"Category '{best['category']}' from {best['file']}/{best['sheet']}:\n"
#             f"Last observed: {df_prophet['y'].iloc[-1]} on {df_prophet['ds'].iloc[-1].date()}\n"
#             "Forecast:\n" +
#             "\n".join(
#                 f"{row['ds'].date()}: {row['yhat']:.2f}"
#                 for _, row in fc_filt.iterrows()
#             )
#         )
#         insight_text = gen("You are a business consultant.", summary + "\n\nProvide actionable insights.")

#         # Save insights (include sources)
#         ins_path = os.path.join(insights, f"{best['category']}_insights.txt")
#         with open(ins_path, "w", encoding="utf-8") as f:
#             f.write("Sources:\n" + "\n".join(sources) + "\n\n")
#             f.write(insight_text)
#         print("Insights saved:", ins_path)

#     # 7b) Otherwise → text‑only insights
#     else:
#         combined = "\n\n".join(d["text"] for d in text_docs)
#         prompt = (
#             f"Sources: {', '.join(sources)}\n\n"
#             f"Relevant content:\n{combined}\n\n"
#             f"User query: {user_query}\n\n"
#             "Provide actionable insights and recommendations."
#         )
#         insight_text = gen("You are a business consultant.", prompt)

#         # Timestamped filename to avoid overwrite
#         ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         out = os.path.join(other_res, f"other_insights_{ts}.txt")
#         with open(out, "w", encoding="utf-8") as f:
#             f.write(insight_text)
#         print("Text-only insights saved:", out)


# if __name__ == "__main__":
#     main()
####################################################################################
# import os
# import json
# import datetime
# from src import etl, analysis, llm, file_processor, insights_other

# def main():
#     # -----------------------------
#     # 1) Folder setup
#     # -----------------------------
#     excel_in  = "data/input"
#     excel_mid = "data/intermediate"
#     excel_out = "data/output"
#     graphs_folder        = "data/graphs"
#     insights_folder      = "data/insights"
#     input_other_folder   = "data/input_other"
#     other_text_folder    = "data/other_text"
#     other_results_folder = "data/other_results"

#     for folder in [
#         excel_in, excel_mid, excel_out,
#         graphs_folder, insights_folder,
#         input_other_folder, other_text_folder, other_results_folder
#     ]:
#         os.makedirs(folder, exist_ok=True)

#     # -----------------------------
#     # 2) ETL Excel → JSON
#     # -----------------------------
#     # Uncomment the next line to run your ETL step
#     # etl.run_etl_pipeline(excel_in, excel_mid, excel_out)

#     # -----------------------------
#     # 3) Build FAISS index for Excel data
#     # -----------------------------
#     index, documents, embedder = analysis.build_faiss_index(excel_out)

#     # -----------------------------
#     # 4) LLM setup (Ollama)
#     # -----------------------------
#     config = llm.load_config("config/config.json")
#     _, _, _ = llm.authenticate_and_load_model(config)
#     generate_response = lambda sys, usr, **kwargs: llm.generate_response(sys, usr, **kwargs)


#     # -----------------------------
#     # 5) User query
#     # -----------------------------
#     user_query = input("Enter your query: ").strip()

#     # -----------------------------
#     # 6) Try Excel first
#     # -----------------------------
#     most_relevant_category = analysis.analyze_query_with_llm_faiss(
#         user_query, generate_response, index, documents, embedder
#     )
#     print("Most relevant Excel category:", most_relevant_category)

#     # Load all Excel JSON files
#     json_files = analysis.find_json_files(excel_out)
#     def extract_values(files, category):
#         vals = []
#         for fp in files:
#             with open(fp, 'r', encoding='utf-8') as f:
#                 content = json.load(f)
#             data = content.get("data", {})
#             if category in data:
#                 vals.extend(data[category])
#         return vals

#     excel_values = extract_values(json_files, most_relevant_category)

#     # -----------------------------
#     # 7) If not enough Excel data → other files branch
#     # -----------------------------
#     if len(excel_values) < 2:
#         print("Insufficient Excel data. Processing other file types...")

#         # 7a) Extract text from each file in input_other_folder
#         for fname in os.listdir(input_other_folder):
#             path = os.path.join(input_other_folder, fname)
#             text = file_processor.extract_text_from_file(path)
#             if not text:
#                 continue
#             out_txt = os.path.splitext(fname)[0] + ".txt"
#             with open(os.path.join(other_text_folder, out_txt), "w", encoding="utf-8") as f:
#                 f.write(text)

#         # 7b) Load all extracted texts
#         other_texts = insights_other.load_other_texts(other_text_folder)
#         sources = list(other_texts.keys())

#         # 7c) Generate insights via LLM
#         other_insights = insights_other.generate_other_insights(
#             user_query, other_texts, generate_response
#         )

#         # 7d) Save timestamped output
#         ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         out_file = os.path.join(other_results_folder, f"other_insights_{ts}.txt")
#         with open(out_file, "w", encoding="utf-8") as f:
#             f.write(f"User Query: {user_query}\n")
#             f.write("Sources:\n" + "\n".join(sources) + "\n\n")
#             f.write("Generated Insights:\n" + other_insights)
#         print(f"Other‑files insights saved at: {out_file}")

#     # -----------------------------
#     # 8) Otherwise → Excel branch (unchanged)
#     # -----------------------------
#     else:
#         # 8a) Plot & forecast
#         graph_file = os.path.join(graphs_folder, f"{most_relevant_category}_trend.png")
#         analysis.plot_trend_and_save(excel_values, most_relevant_category, graph_file)
#         print(f"Graph saved at: {graph_file}")

#         # 8b) Display predicted values
#         analysis.display_predicted_values(excel_values)

#         # 8c) Generate insights for Excel data
#         insights_text = analysis.generate_insights_from_predictions(
#             excel_values, most_relevant_category, generate_response
#         )
#         insights_file = os.path.join(insights_folder, f"{most_relevant_category}_insights.txt")
#         with open(insights_file, "w", encoding="utf-8") as f:
#             f.write(insights_text)
#         print(f"Insights saved at: {insights_file}")

# if __name__ == "__main__":
#     main()

###############3mmclarifyxl$$$$$$$$$$$$$$$$$$$$$$$$##################


# import os
# import json
# import datetime
# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt

# from src import etl, analysis, llm, file_processor, insights_other

# def main():
#     # 1) Paths
#     input_folder          = "data/input"
#     intermediate_folder   = "data/intermediate"
#     output_folder         = "data/output"
#     graphs_folder         = "data/graphs"
#     insights_folder       = "data/insights"
#     input_other_folder    = "data/input_other"
#     other_text_folder     = "data/other_text"
#     other_results_folder  = "data/other_results"

#     # Ensure directories exist
#     for d in [
#         input_folder, intermediate_folder, output_folder,
#         graphs_folder, insights_folder,
#         input_other_folder, other_text_folder, other_results_folder
#     ]:
#         os.makedirs(d, exist_ok=True)

#     # 2) ETL Excel → JSON (uncomment to run)
#     # etl.run_etl_pipeline(input_folder, intermediate_folder, output_folder)

#     # 3) Build FAISS index for Excel data
#     index, documents, embedder = analysis.build_faiss_index(output_folder)

#     # 4) Load LLM config & authenticate (Ollama)
#     config = llm.load_config("config/config.json")
#     _, _, _ = llm.authenticate_and_load_model(config)
#     generate_response = lambda sys, usr, temperature=0.7, max_tokens=500: \
#         llm.generate_response(sys, usr, temperature, max_tokens)

#     # 5) Ask the user
#     user_query = input("Enter your query: ").strip()

#     # 6) Find best Excel category
#     best_cat = analysis.analyze_query_with_llm_faiss(
#         user_query, generate_response, index, documents, embedder
#     )
#     print("Most relevant Excel category:", best_cat)

#     # 7) Load all JSONs and extract values for that category
#     json_files = analysis.find_json_files(output_folder)
#     def extract_values(files, category):
#         vals = []
#         for fp in files:
#             with open(fp, 'r', encoding='utf-8') as f:
#                 content = json.load(f)
#             data = content.get("data", {})
#             if category in data:
#                 vals.extend(data[category])
#         return vals

#     excel_values = extract_values(json_files, best_cat)

#     # 8) Retrieve relevant docs for multi-modal context
#     retrieved = analysis.retrieve_relevant_docs(user_query, index, documents, embedder, top_k=5)

#     # 9) Build 'sources' list robustly
#     sources = []
#     for d in retrieved:
#         if "metadata" in d:
#             m = d["metadata"]
#             fn = m.get("file_name", "<unknown>")
#             sh = m.get("sheet_name")
#             sources.append(f"{fn}/{sh}" if sh else fn)
#         elif d.get("type") == "timeseries":
#             sources.append(f"{d['file']}/{d['sheet']}")
#         else:
#             sources.append(d.get("file", "<unknown>"))
#     seen = set()
#     sources = [s for s in sources if not (s in seen or seen.add(s))]

#     # Print sources to console
#     print("=== Files used to generate insights ===")
#     for src in sources:
#         print(" •", src)
#     print("=======================================")

#     # 10) Branch on whether we have enough Excel data
    
#     if len(excel_values) >= 2:
#         # ----- Excel + Multi‑Modal path -----

#         # a) Build DataFrame from excel_values
#         df_ts = pd.DataFrame(excel_values)
#         df_ts['Date'] = pd.to_datetime(df_ts['Date'])
#         df_ts = df_ts.sort_values('Date')
#         df_prophet = df_ts.rename(columns={'Date': 'ds', 'value': 'y'})

#         # b) Forecast with Prophet
#         model = Prophet()
#         model.fit(df_prophet)
#         future = model.make_future_dataframe(periods=14, freq='M')
#         fc = model.predict(future)
#         last_ds = df_prophet['ds'].max()
#         fc_filt = fc[fc['ds'] > last_ds]

#         # c) Plot observed + forecast
#         plt.figure(figsize=(10, 6))
#         plt.plot(df_prophet['ds'], df_prophet['y'], label='Observed', linewidth=2)
#         plt.plot(fc_filt['ds'], fc_filt['yhat'], '--', label='Forecast', linewidth=2)
#         if not fc_filt.empty:
#             plt.plot(
#                 [df_prophet['ds'].iloc[-1], fc_filt['ds'].iloc[0]],
#                 [df_prophet['y'].iloc[-1], fc_filt['yhat'].iloc[0]],
#                 '--'
#             )
#         plt.title(f'Trend & Forecast: {best_cat}')
#         plt.xlabel('Date')
#         plt.ylabel(best_cat)
#         plt.legend()
#         plt.tight_layout()

#         graph_file = os.path.join(graphs_folder, f"{best_cat}_trend.png")
#         plt.savefig(graph_file)
#         plt.close()
#         print("Graph saved at:", graph_file)

#         # d) Save graph metadata
#         meta = {
#             "generated_at": datetime.datetime.now().isoformat(),
#             "category": best_cat,
#             "sources": sources
#         }
#         with open(graph_file.replace('.png', '.meta.json'), 'w', encoding='utf-8') as mf:
#             json.dump(meta, mf, indent=2)

#         # e) Pull in any retrieved “text” docs
#         other_context = ""
#         text_docs = [d for d in retrieved if d.get("type") == "text"]
#         for d in text_docs:
#             fname = d["file"]
#             txt_path = os.path.join(other_text_folder, fname)
#             if os.path.exists(txt_path):
#                 with open(txt_path, 'r', encoding='utf-8') as tf:
#                     snippet = tf.read()
#                 other_context += f"\n--- Context from {fname} ---\n{snippet}\n"

#         # f) Build the full LLM prompt
#         summary = (
#             f"Category '{best_cat}' from Excel.\n"
#             f"Last observed: {df_prophet['y'].iloc[-1]} on {df_prophet['ds'].iloc[-1].date()}.\n"
#             "Forecast:\n" +
#             "\n".join(f"{row['ds'].date()}: {row['yhat']:.2f}"
#                       for _, row in fc_filt.iterrows())
#         )
#         full_prompt = (
#             summary
#             + "\n\nAdditional context from other files:" + other_context
#             + "\n\nPlease provide actionable insights and recommendations based on all of the above."
#         )
#         insight_text = generate_response("You are a business consultant.", full_prompt)

#         # g) Save insights file
#         insights_file = os.path.join(insights_folder, f"{best_cat}_insights.txt")
#         with open(insights_file, 'w', encoding='utf-8') as outf:
#             outf.write("User Query:\n" + user_query + "\n\n")
#             outf.write("Sources:\n" + "\n".join(sources) + "\n\n")
#             outf.write("Insights:\n" + insight_text)
#         print("Insights saved at:", insights_file)



# if __name__ == "__main__":
#     main()

###################display all laSt########################################################################################################

# import os
# import json
# import datetime

# import pandas as pd
# import matplotlib.pyplot as plt
# from prophet import Prophet

# from src import etl, analysis, llm, file_processor, insights_other

# def main():
#     # 1) Define folders
#     excel_in            = "data/input"
#     excel_mid           = "data/intermediate"
#     excel_out           = "data/output"
#     graphs_folder       = "data/graphs"
#     insights_folder     = "data/insights"
#     other_in            = "data/input_other"
#     other_text_folder   = "data/other_text"
#     other_results_folder= "data/other_results"

#     for d in [excel_in, excel_mid, excel_out,
#               graphs_folder, insights_folder,
#               other_in, other_text_folder, other_results_folder]:
#         os.makedirs(d, exist_ok=True)

#     # 2) (Optional) Run Excel ETL
#     # etl.run_etl_pipeline(excel_in, excel_mid, excel_out)

#     # 3) Extract all other file text
#     for fname in os.listdir(other_in):
#         path = os.path.join(other_in, fname)
#         txt = file_processor.extract_text_from_file(path)
#         if not txt:
#             continue
#         out_txt = os.path.splitext(fname)[0] + ".txt"
#         with open(os.path.join(other_text_folder, out_txt), "w", encoding="utf-8") as f:
#             f.write(txt)

#     # 4) Build unified FAISS index over Excel JSON & extracted text
#     index, documents, embedder = analysis.build_unified_index(
#         excel_out, other_text_folder
#     )

#     # 5) LLM setup (Ollama)

#     ##multimodal

#     # config = llm.load_config("config/config.json")
#     # _, _, _ = llm.authenticate_and_load_model(config)
#     # generate_response = lambda sys, usr, temperature=0.7, max_tokens=500: \
#     #     llm.generate_response(sys, usr, temperature, max_tokens)

#     ####aiagents
#     config = llm.load_config("config/config.json")
#     llm.authenticate_and_load_model(config)
#     generate_response = lambda sys, usr, temperature=0.7, max_tokens=500: llm.generate_response(sys, usr)


#     # 6) Get user query
#     user_query = input("Enter your query: ").strip()

#     # 7) Retrieve top-5 relevant docs (timeseries + text)
#     retrieved = analysis.retrieve_relevant_docs(
#         user_query, index, documents, embedder, top_k=5
#     )

#     # 8) Collect & print source filenames
#     sources = []
#     for d in retrieved:
#         if d["type"] == "timeseries":
#             sources.append(f"{d['file']}/{d['sheet']}")
#         else:
#             sources.append(d["file"])
#     # dedupe
#     seen = set()
#     sources = [s for s in sources if not (s in seen or seen.add(s))]

#     print("\n=== Files used to generate insights ===")
#     for s in sources:
#         print(" •", s)
#     print("=======================================\n")

#     # 9) Extract Excel values for the best category
#     #    (top timeseries hit)
#     ts_hits = [d for d in retrieved if d["type"]=="timeseries"]
#     if ts_hits and len(ts_hits[0]["series"]) >= 2:
#         best = ts_hits[0]
#         excel_values = best["series"]

#         # a) Prepare DataFrame
#         df_ts = pd.DataFrame(excel_values)
#         df_ts['Date'] = pd.to_datetime(df_ts['Date'])
#         df_ts = df_ts.sort_values('Date')
#         df_prophet = df_ts.rename(columns={'Date':'ds','value':'y'})

#         # b) Forecast
#         model = Prophet()
#         model.fit(df_prophet)
#         future = model.make_future_dataframe(periods=14, freq='M')
#         fc = model.predict(future)
#         last_ds = df_prophet['ds'].max()
#         fc_filt = fc[fc['ds'] > last_ds]

#         # c) Plot
#         plt.figure(figsize=(10,6))
#         plt.plot(df_prophet['ds'], df_prophet['y'], label='Observed', linewidth=2)
#         plt.plot(fc_filt['ds'], fc_filt['yhat'], '--', label='Forecast', linewidth=2)
#         if not fc_filt.empty:
#             plt.plot(
#                 [df_prophet['ds'].iloc[-1], fc_filt['ds'].iloc[0]],
#                 [df_prophet['y'].iloc[-1], fc_filt['yhat'].iloc[0]],
#                 '--'
#             )
#         plt.title(f"Trend & Forecast: {best['category']}")
#         plt.xlabel('Date')
#         plt.ylabel(best['category'])
#         plt.legend()
#         plt.tight_layout()

#         graph_file = os.path.join(graphs_folder, f"{best['category']}_trend.png")
#         plt.savefig(graph_file)
#         plt.close()
#         print("Graph saved at:", graph_file)

#         # d) Gather additional text context
#         other_context = ""
#         for d in retrieved:
#             if d["type"] == "text":
#                 txtpath = os.path.join(other_text_folder, d["file"])
#                 snippet = open(txtpath, 'r', encoding='utf-8').read()
#                 other_context += f"\n--- {d['file']} ---\n{snippet}\n"

#         # e) Build full prompt and call LLM
#         summary = (
#             f"Category '{best['category']}' from {best['file']}/{best['sheet']}\n"
#             f"Last observed: {df_prophet['y'].iloc[-1]} on {df_prophet['ds'].iloc[-1].date()}\n"
#             "Forecast:\n" +
#             "\n".join(f"{row['ds'].date()}: {row['yhat']:.2f}"
#                       for _, row in fc_filt.iterrows())
#         )
#         full_prompt = summary + "\n\nAdditional context:\n" + other_context \
#                       + "\n\nProvide actionable insights."
#         insight_text = generate_response("You are a business consultant.", full_prompt)

#         # f) Save insights
#         out_path = os.path.join(insights_folder, f"{best['category']}_insights.txt")
#         with open(out_path, 'w', encoding='utf-8') as f:
#             f.write("User Query:\n" + user_query + "\n\n")
#             f.write("Sources:\n" + "\n".join(sources) + "\n\n")
#             f.write("Insights:\n" + insight_text)
#         print("Insights saved at:", out_path)

#     else:
#         # ----- Text‑only fallback -----
#         print("No usable time-series → generating text-only insights")

#         # a) Combine all retrieved text docs
#         combined = ""
#         for d in retrieved:
#             if d["type"] == "text":
#                 txtpath = os.path.join(other_text_folder, d["file"])
#                 combined += f"\n--- {d['file']} ---\n" \
#                             + open(txtpath, 'r', encoding='utf-8').read()

#         prompt = (
#             f"User query: {user_query}\n\n"
#             f"Relevant content:\n{combined}\n\n"
#             "Please provide actionable insights based on the above."
#         )
#         other_insights = generate_response("You are a business consultant.", prompt)
#         ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         out_file = os.path.join(other_results_folder, f"other_insights_{ts}.txt")
#         with open(out_file, 'w', encoding='utf-8') as f:
#             f.write("User Query:\n" + user_query + "\n\n")
#             f.write("Sources:\n" + "\n".join(sources) + "\n\n")
#             f.write("Insights:\n" + other_insights)
#         print("Text‑only insights saved at:", out_file)

# if __name__ == "__main__":

#     main()


#########agents

# # src/main.py
# from src.agents import agent

# def main():
#     print("🎯 Business Insights Generator (agentic mode)\n")
#     query = input("Enter your business-insight query: ").strip()
#     result = agent.run(query)
#     print("\n=== Agent Response ===")
#     print(result)

# if __name__ == "__main__":
#     main()

##newdata

import os, datetime
import pandas as pd
from prophet import Prophet

from src import analysis, llm

def main():
    # --- Folders ---
    json_root    = "data/json_input"   # top-level folder containing your 4 subfolders
    graphs_folder = "data/graphs"
    insights_folder = "data/insights"
    os.makedirs(graphs_folder, exist_ok=True)
    os.makedirs(insights_folder, exist_ok=True)

    # --- LLM setup ---
    config = llm.load_config("config/config.json")
    _, _, _ = llm.authenticate_and_load_model(config)
    gen = lambda sys, usr: llm.generate_response(sys, usr)

    # --- 1) Build product index from all JSONs ---
    index, docs, embedder = analysis.build_product_index(json_root)

    # --- 2) Get user’s query ---
    query = input("Enter your query (e.g. product name): ").strip()

    # --- 3) Find best matching product ---
    sys_p = "You are a product-search assistant."
    refined = gen(sys_p, query)
    # reuse your FAISS search fn:
    best = analysis.search_vector_store(refined, index, docs, embedder, top_k=1)[0]
    product = best["product"]
    print("Most relevant product:", product, "in file", best["file"])

    # --- 4) Extract its time series from all JSONs ---
    series = analysis.extract_product_timeseries(json_root, product)

    # --- 5) Plot & forecast if we have ≥2 points ---
    if len(series) >= 2:
        df = pd.DataFrame(series).rename(columns={"Date":"ds","value":"y"})
        model = Prophet().fit(df)
        future = model.make_future_dataframe(periods=14, freq='M')
        fc = model.predict(future)
        fc = fc[fc.ds > df.ds.max()]

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.plot(df.ds, df.y, label="Observed")
        plt.plot(fc.ds, fc.yhat, "--", label="Forecast")
        plt.legend()
        graph_path = os.path.join(graphs_folder, f"{product}_trend.png")
        plt.savefig(graph_path); plt.close()
        print("Graph saved to", graph_path)

        # --- 6) Generate insights over the forecast ---
        # insights = analysis.generate_insights_from_predictions(fc[['ds','yhat']].rename(
        #     columns={'ds':'Date','yhat':'value'}), product, gen)
        # ✅ pass raw prophet output
        insights = analysis.generate_insights_from_predictions(
            fc[['ds','yhat']],
            product,
            gen
        )



    else:
        print("Not enough time series data for", product)
        insights = f"No time-series ≥2 points found for {product}."

    # --- 7) Save one insights file ---
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(insights_folder, f"insights_{ts}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\nProduct: {product}\n\n")
        f.write(insights)
    print("Insights saved to", out)


if __name__=="__main__":
    main()
