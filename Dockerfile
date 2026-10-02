FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_stonks/ agent_stonks/
COPY main.py run_app.py ./
# Streamlit settings the app relies on (the message cache off; see README, "Recovery").
COPY .streamlit/config.toml .streamlit/config.toml

EXPOSE 8501

# run_app.py restarts the app on a crash or a hang (see README, "Recovery").
CMD ["python", "run_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
