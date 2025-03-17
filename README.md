# research-engineering-intern-assignment

🚀 Misinformation Detection & Social Media Analysis
===================================================
This project is a machine learning-powered misinformation detection system, featuring:
✅ A FastAPI backend to classify Reddit posts as misinformation or reliable  
✅ A Streamlit dashboard for data visualization and interactive analysis   

📌 Features
-----------
1️⃣ Misinformation Detection: Uses an LLM to analyze Reddit posts  
2️⃣ Time-Series Analysis: Tracks trending discussions over time  
3️⃣ Subreddit Analysis: Identifies top contributors to discussions  
4️⃣ Network Visualization: Shows connections between common keywords  

⚙️ Installation Guide
----------------------
1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Misinfo_detect.git
cd Misinfo_detect
```

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Download NLP Resources
```bash
python -m nltk.downloader stopwords
```

🚀 Running the Project Locally
------------------------------
1️⃣ Start the FastAPI Backend
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
- The API should now be running at `http://127.0.0.1:8000`  
- Test the API:
```bash
curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"text": "This is a test Reddit post."}'
```

2️⃣ Start the Streamlit Dashboard
```bash
streamlit run dashboard.py
```
- The dashboard will open in your browser.

🛠 API Documentation
---------------------
The FastAPI model is deployed as a REST API.

📝 Endpoint: `POST /predict/`
**URL:**  
```
https://your-api.onrender.com/predict/
```
**Request Body (JSON):**
```json
{
  "text": "This is a Reddit post to check for misinformation."
}
```
**Response (JSON):**
```json
{
  "prediction": "Misinformation"
}
```


📊 Visualizations in Dashboard
------------------------------
1️⃣ Time-Series Graph: Posts matching a search query  
2️⃣ Trending Topics: Topic modeling using LDA  
3️⃣ Top Subreddits: Pie chart of most active communities  
4️⃣ Keyword Network: Graph visualization of shared terms  

📌 Future Improvements
----------------------
🔹 Improve LLM Model Accuracy (More training data)  
🔹 Expand to Other Social Media (Twitter, Facebook, etc.)  
🔹 Advanced NLP Features (Sentiment analysis, Named Entity Recognition)  

💡 Contributors
---------------
👨‍💻 **Krishang Goel**  
📧 **krishang.229309035@muj.manipal.edu**  

