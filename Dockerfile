# Customer Support Triage — OpenEnv Environment
# Works with: docker build . && docker run -p 7860:7860 <image>

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY server/ ./server/

# Environment defaults (override at runtime)
ENV SUPPORT_TASK=easy
ENV HOST=0.0.0.0
ENV PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.app:app --host $HOST --port $PORT"]
