# cu (Desktop Agent & UI Tools)

This repository contains a fullstack desktop agent with a web UI, VNC integration, and automated UI testing tools.

## Requirements

- Python 3.8+
- Node.js 18+
- Linux (tested on Arch Linux)
- System binaries: `Xvfb`, `openbox`, `x11vnc`, `scrot`, `xdotool`, `ss`, `xdpyinfo`

## Python Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Ensure required system binaries are installed:**
   ```bash
   sudo pacman -S xorg-server-xvfb openbox x11vnc scrot xdotool iproute2 xorg-xdpyinfo
   # or use your distro's package manager
   ```
3. **Run the agent:**
   ```bash
   python3 onefile_fullstack_agent.py
   ```
   The web UI will be available at [http://127.0.0.1:7860](http://127.0.0.1:7860)

## Node.js UI Testing Tools

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```
2. **Run automated UI tests:**
   ```bash
   npm run test
   # or
   node run_agent_ui_test.mjs
   ```

## Project Structure

- `onefile_fullstack_agent.py` — Main Python agent and web server
- `run_agent_ui_test.mjs`, `test_agent_ui.mjs` — Node.js Puppeteer-based UI tests
- `debug-chat-socket.js` — Debugging tool for Socket.IO/WebSocket events

## GitHub Repository

Clone and install:
```bash
git clone https://github.com/nm-z/cu.git
cd cu
pip install -r requirements.txt
npm install
```

## Notes
- The agent expects LM Studio to be running at `http://127.0.0.1:1234/v1` by default. You can override this with the `LMSTUDIO_URL` environment variable.
- For VNC viewing, connect to `127.0.0.1:12345` (default port).
- For development or troubleshooting, check `agent_stderr.log` and `agent_stdout.log`. 