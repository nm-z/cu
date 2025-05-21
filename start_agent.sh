#!/usr/bin/env bash
set -e

# Find and kill process using port 7860
PID=$(lsof -t -i:7860 || true)
if [ -n "$PID" ]; then
  echo "Killing process on port 7860 (PID $PID)"
  kill $PID
  sleep 2
fi

# Start the agent in the background
nohup python3 onefile_fullstack_agent.py > agent_stdout.log 2> agent_stderr.log &
AGENT_PID=$!
echo "Started agent with PID $AGENT_PID"

# Tail logs for quick debugging
sleep 2
echo "--- agent_stderr.log (last 40 lines) ---"
tail -40 agent_stderr.log || true
echo "--- agent_stdout.log (last 40 lines) ---"
tail -40 agent_stdout.log || true 