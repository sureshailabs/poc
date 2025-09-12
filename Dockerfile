# Use a small, modern Python base
FROM python:3.10.9

# System settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

# Install minimal OS deps (certs for HTTPS calls to LLM/ClickHouse)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + optional schema file
COPY main.py /app/
# If you keep the DDL in the image, uncomment next line:
COPY orbit_japa.sql /app/orbit_japa.sql


EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
