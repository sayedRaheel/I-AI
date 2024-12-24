FROM runpod/pytorch:2.0.1-py3.10-cuda11.7.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY handler.py .
COPY sayed_voice.json .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the handler
CMD [ "python", "-u", "handler.py" ]