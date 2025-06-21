# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import json
# import random
# from datetime import datetime

# app = FastAPI(title="Mining Data API")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load the historical data
# def load_historical_data():
#     try:
#         with open('power-historic-data-api.py', 'r') as f:
#             content = f.read()
#             # Extract JSON data (skip the comments at the top)
#             json_start = content.find('{')
#             json_content = content[json_start:]
#             return json.loads(json_content)
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return {"data": []}

# # Load data once at startup
# historical_data = load_historical_data()

# @app.get("/")
# def read_root():
#     return {"message": "Mining Data API", "endpoints": ["/mining/live", "/mining/historical"]}

# @app.get("/mining/live")
# def get_live_data():
#     """Return a random data point from historical data to simulate live data"""
#     if historical_data and "data" in historical_data:
#         # Get a random data point and update timestamp to current time
#         data_point = random.choice(historical_data["data"]).copy()
#         data_point["timestamp"] = datetime.now().isoformat()
#         return data_point
#     else:
#         return {
#             "error": "No data available",
#             "timestamp": datetime.now().isoformat()
#         }

# @app.get("/mining/historical")
# def get_historical_data():
#     """Return all historical data"""
#     return historical_data

# @app.get("/mining/metadata")
# def get_metadata():
#     """Return metadata about the mining facility"""
#     if historical_data and "metadata" in historical_data:
#         return historical_data["metadata"]
#     else:
#         return {"error": "No metadata available"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000) 