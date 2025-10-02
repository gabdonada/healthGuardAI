# healthGuardAI
This is a LLM that will answer your questions related to Midea HealthGuard as the manual.
It answers only in Portuguese Brazil and doesn't work with context (just a simple API).

Install on our notebook:
`pip install PyPDF2==3.0.1 sentence-transformers==2.2.2 faiss-cpu==1.7.4 transformers==4.34.0 accelerate==0.21.0 langchain==0.0.300 tiktoken gradio boto3`

Tu run the API you should run
`uvicorn handler:app --host 0.0.0.0 --port 8000 --reload`

Request to API
`curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question":"your question goes here"}'`
