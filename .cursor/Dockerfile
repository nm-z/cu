# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    xvfb openbox x11vnc scrot xdotool iproute2 xdpyinfo \
    nodejs npm curl git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy Python files
COPY onefile_fullstack_agent.py requirements.txt ./

# Copy Node.js files
COPY package.json ./
COPY run_agent_ui_test.mjs test_agent_ui.mjs debug-chat-socket.js ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
RUN npm install

# Copy the rest of the files (README, etc.)
COPY README.md LICENSE ./

# Expose the web UI port
EXPOSE 7860

# Set environment variables for LM Studio and display
ENV LMSTUDIO_URL=http://127.0.0.1:1234/v1 \
    DISPLAY=:99 \
    RFB_PORT=12345 \
    UI_PORT=7860

# Default command: run the agent
CMD ["python3", "onefile_fullstack_agent.py"]
