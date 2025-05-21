#!/usr/bin/env python3
"""
Desktop Agent: A full-stack desktop automation tool with AI control.

This module provides a complete desktop automation solution with:
- Desktop interaction tools (mouse, keyboard, screenshot, shell commands)
- Real filesystem-based file editing tools
- Headless X server with VNC remote access
- Flask/SocketIO-based web UI for chat and monitoring
- Integration with OpenAI-compatible LLM APIs

All components run in a single Python process for simplicity and portability.
"""

# Disable eventlet to avoid context and monkey-patching issues; use threading only
_EVENTLET_AVAILABLE = False

# Standard library imports
import asyncio
import atexit
import base64
import importlib
import json
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Third-party imports
import nest_asyncio
import openai
import requests
import websockets
from flask import Flask, jsonify, render_template_string
from flask_socketio import SocketIO

# ─────────────── Global Logger for Script ───────────────────
# Configure this early so all parts of the script can use it.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)  # Main logger for the application

# ─────────────── Fail-Fast Setup ─────────────────────────────
def fatal(msg: str, *args, exc_info=False):
    """Log a critical error and exit the program.

    Args:
        msg: The message format string
        *args: Variable arguments to format the message
        exc_info: Whether to include exception info in the log
    """
    formatted_msg = msg % args if args else msg
    logger.critical(formatted_msg, exc_info=exc_info)
    sys.exit(1)  # Use sys.exit to allow atexit handlers to run

def on_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions in the main thread."""
    if issubclass(exc_type, KeyboardInterrupt):
        logger.warning("KeyboardInterrupt received. Exiting gracefully...")
        sys.exit(0)  # Normal exit for Ctrl+C
    else:
        # Use logger, not fatal, to avoid recursion if logger itself fails
        logging.getLogger("CRITICAL_ERROR_HOOK").critical(
            "Uncaught exception in main thread:",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        os._exit(1)  # Hard exit for other uncaught exceptions in main thread

sys.excepthook = on_uncaught_exception

def asyncio_loop_exception_handler(loop, context):
    """Handle uncaught exceptions in the asyncio event loop."""
    msg = context.get("exception", context.get("message", "Unknown asyncio error"))
    # Use logger, not fatal, to avoid recursion if logger itself fails
    logging.getLogger("CRITICAL_ASYNC_ERROR_HOOK").critical(
        "Uncaught exception in asyncio event loop: %s", msg,
        exc_info=context.get("exception")
    )
    os._exit(1)  # Hard exit

# Wrapper for creating threads that will call fatal() on unhandled exceptions
def crash_on_thread_exception(target_func, *args, **kwargs):
    """Create a thread that will call fatal() on unhandled exceptions.

    Args:
        target_func: The function to run in the thread
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The started thread
    """
    def wrapper():
        try:
            target_func(*args, **kwargs)
        except SystemExit as e:  # Allow sys.exit() to propagate if called explicitly
            raise e
        except KeyboardInterrupt:
            logger.warning(
                "KeyboardInterrupt in thread %s. Exiting thread.",
                threading.current_thread().name
            )
            # Let main thread handler do the full exit/cleanup for KeyboardInterrupt
        except Exception as e:
            fatal(
                "Unhandled exception in thread %s: %s",
                threading.current_thread().name, e,
                exc_info=True
            )

    thread = threading.Thread(target=wrapper, daemon=True)  # Ensure daemon=True
    thread.start()
    return thread

# ───────────────  USER CONFIG  ────────────────
LMSTUDIO_BASE = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
MODEL_NAME = os.environ.get("LMSTUDIO_MODEL", "qwen2.5-vl-7b-instruct")
DISPLAY = os.environ.get("DISPLAY", ":99")          # Xephyr/Xvfb seat
RFB_PORT = int(os.environ.get("RFB_PORT", "12345"))  # VNC
UI_PORT = int(os.environ.get("UI_PORT", "7860"))    # UI Port
# ───────────────────────────────────────────────

# ---------- minimal OpenAI-compatible client ----------
openai.api_key  = "dummy"
openai.api_base = LMSTUDIO_BASE

nest_asyncio.apply() # Apply the patch

# Set asyncio exception handler *after* nest_asyncio.apply() and getting the loop
loop = asyncio.get_event_loop() # Get the event loop that nest_asyncio might have patched
loop.set_exception_handler(asyncio_loop_exception_handler)

# ─────────────── Pre-Flight Checks ───────────────────────────
def pre_flight_checks():
    """Run pre-flight checks to ensure all dependencies are available."""
    logger.info("Running pre-flight checks...")

    # 1. Python imports (critical ones for core functionality)
    try:
        import flask
        import flask_socketio
        logger.info("✔︎ Critical Python dependencies seem to be imported.")
    except ImportError as e:
        fatal("Missing critical Python dependency: %s", e)

    # 2. Binaries on PATH
    required_binaries = ("Xvfb", "openbox", "x11vnc", "scrot", "xdotool", "ss", "xdpyinfo")
    for prog in required_binaries:
        if shutil.which(prog) is None:
            fatal("Required binary not found on PATH: %s", prog)
    logger.info("✔︎ All required binaries found: %s", ", ".join(required_binaries))

    # 3. LM Studio API reachable
    if not LMSTUDIO_BASE.startswith("http"):  # Basic check
        fatal("LMSTUDIO_BASE URL does not look valid: %s", LMSTUDIO_BASE)
    try:
        logger.info("Checking LM Studio API at %s/models ...", LMSTUDIO_BASE)
        # Use a session for potential keep-alive and header management
        with requests.Session() as req_session:
            r = req_session.get(f"{LMSTUDIO_BASE}/models", timeout=5)
        r.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        logger.info("✔︎ LM Studio API reachable at %s and responded successfully.", LMSTUDIO_BASE)
    except requests.exceptions.RequestException as e:
        fatal("Cannot reach or get a valid response from LMStudio API at %s: %s", LMSTUDIO_BASE, e)
    except Exception as e:  # Catch other potential errors like JSONDecodeError if we check content
        fatal("Error during LMStudio API check at %s: %s", LMSTUDIO_BASE, e)

    logger.info("Pre-flight checks passed.")

# Force DISPLAY to :99, overriding any external environment variable for script execution
os.environ["DISPLAY"] = ":99"
DISPLAY = ":99" # Also update the global constant

# --- Flask App Setup ---
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret_agent_key!'  # Replace with a real secret in production
# Using async_mode='threading' as our agent/tool calls are async via asyncio,
# but Flask itself is sync. SocketIO can bridge this.
# Consider 'asgi' with a suitable ASGI server (like Uvicorn) if full async stack is desired later.
sio = SocketIO(app, cors_allowed_origins="*")

CHAT_HISTORY = []  # Global chat history

# --- VNC Proxying Setup ---
# We'll use a simple target for x11vnc, assuming it's on localhost
VNC_SERVER_HOST = '127.0.0.1'
VNC_SERVER_PORT = RFB_PORT
VNC_WS_PROXY_PORT = 12346  # New WebSocket proxy port for noVNC

# --- New WebSocket VNC Proxy using 'websockets' ---
# Robust VNC WebSocket proxy with proper event loop management and error handling
class VNCWebSocketProxy:
    """WebSocket proxy for VNC connections."""

    def __init__(self, ws_host, ws_port, vnc_host, vnc_port):
        """Initialize the VNC WebSocket proxy."""
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.vnc_host = vnc_host
        self.vnc_port = vnc_port
        self.server = None
        self.stop_event = asyncio.Event()
        self.log("Initializing VNC WebSocket proxy %s:%s -> %s:%s", 
                ws_host, ws_port, vnc_host, vnc_port)

    def log(self, msg, *args):
        """Log with a prefix."""
        logger.info("[VNC WS PROXY] " + msg, *args)

    async def handler(self, websocket, path=None):
        """Handle WebSocket connections and proxy them to the VNC server."""
        client_addr = websocket.remote_address if hasattr(websocket, 'remote_address') else 'unknown'
        self.log("New WebSocket connection from %s, path: %s", client_addr, path)
        
        vnc_sock = None
        try:
            # Create a TCP socket to connect to the VNC server
            vnc_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            vnc_sock.settimeout(10)  # Initial connection timeout
            
            # Connect to the VNC server
            try:
                self.log("Connecting to VNC server at %s:%s", self.vnc_host, self.vnc_port)
                vnc_sock.connect((self.vnc_host, self.vnc_port))
                self.log("Connected to VNC server")
                
                # Set to non-blocking mode for async operation
                vnc_sock.setblocking(False)
            except Exception as e:
                self.log("Failed to connect to VNC server: %s", e)
                await websocket.close(1011, f"Failed to connect to VNC: {e}")
                return
            
            # Bidirectional proxy between WebSocket and TCP socket
            # Create tasks for both directions
            ws_to_vnc_task = asyncio.create_task(self._websocket_to_vnc(websocket, vnc_sock))
            vnc_to_ws_task = asyncio.create_task(self._vnc_to_websocket(websocket, vnc_sock))
            
            # Wait for either task to complete
            try:
                done, pending = await asyncio.wait(
                    [ws_to_vnc_task, vnc_to_ws_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel the pending task
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Check if there were any exceptions
                for task in done:
                    exc = task.exception()
                    if exc:
                        self.log("Error in proxy task: %s", exc)
            except asyncio.CancelledError:
                self.log("Handler task cancelled")
                raise
            
        except Exception as e:
            self.log("Error in WebSocket handler: %s", e)
        finally:
            # Clean up
            if vnc_sock:
                try:
                    vnc_sock.close()
                except:
                    pass
            self.log("WebSocket connection closed")

    async def _websocket_to_vnc(self, websocket, vnc_sock):
        """Forward data from WebSocket to VNC."""
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await asyncio.get_event_loop().sock_sendall(vnc_sock, message)
                else:
                    # Text messages are not expected in VNC
                    self.log("Received unexpected text message from WebSocket")
        except Exception as e:
            self.log("Error in WebSocket to VNC forwarding: %s", e)
            await websocket.close(1011, f"Proxy error: {e}")
            raise

    async def _vnc_to_websocket(self, websocket, vnc_sock):
        """Forward data from VNC to WebSocket."""
        try:
            loop = asyncio.get_event_loop()
            while True:
                # Read data from VNC
                data = await loop.sock_recv(vnc_sock, 8192)
                if not data:
                    # Connection closed
                    self.log("VNC server closed connection")
                    break
                
                # Send to WebSocket
                await websocket.send(data)
        except Exception as e:
            self.log("Error in VNC to WebSocket forwarding: %s", e)
            await websocket.close(1011, f"Proxy error: {e}")
            raise

    async def start(self):
        """Start the WebSocket server."""
        try:
            self.log("Starting WebSocket server on %s:%s", self.ws_host, self.ws_port)
            self.stop_event.clear()
            
            # Create server
            self.server = await websockets.serve(
                self.handler, self.ws_host, self.ws_port,
                max_size=None
            )
            
            self.log("WebSocket server started")
            return True
        except Exception as e:
            self.log("Failed to start WebSocket server: %s", e)
            return False

    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.log("Stopping WebSocket server")
            self.stop_event.set()
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            self.log("WebSocket server stopped")
        
    async def is_running(self):
        """Check if the server is running."""
        return self.server is not None and not self.stop_event.is_set()

# Create and start the VNC WebSocket proxy
vnc_ws_proxy = VNCWebSocketProxy('127.0.0.1', VNC_WS_PROXY_PORT, VNC_SERVER_HOST, VNC_SERVER_PORT)
proxy_thread = vnc_ws_proxy.start()

# --- HTML Template for the new UI ---
# Updated to include noVNC client and connect to the new WebSocket backend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Desktop Agent</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: sans-serif;
            display: grid;
            grid-template-columns: 2fr 1fr;
            height: 100vh;
            margin: 0;
            padding: 0;
        }
        #vnc-container {
            border-right: 1px solid #ccc;
            position: relative;
            background-color: #333;
            overflow: hidden;
        }
        #vnc-canvas {
            width: 100%;
            height: 100%;
        }
        #controls-container {
            display: flex;
            flex-direction: column;
            padding: 1rem;
            overflow-y: auto;
        }
        #chat-output, #tool-log {
            flex-grow: 1;
            border: 1px solid #eee;
            padding: .5rem;
            margin-bottom: 10px;
            overflow-y: auto;
            border-radius: .5rem;
        }
        #tool-log {
            height: 200px;
            flex-grow: 0;
        }
        #user-input {
            display: flex;
            margin-top: auto;
        }
        #user-input input {
            flex-grow: 1;
            padding: 8px;
            border-radius: .375rem 0 0 .375rem;
            border: 1px solid #ccc;
        }
        #user-input button {
            padding: 8px;
            background: #4f46e5;
            color: white;
            border: none;
            border-radius: 0 .375rem .375rem 0;
            cursor: pointer;
        }
        .message {
            margin-bottom: 5px;
            padding: 3px;
            border-radius: 3px;
        }
        .user-message {
            background-color: #e1f5fe;
            text-align: right;
        }
        .bot-message {
            background-color: #f0f0f0;
        }
        .system-message {
            background-color: #fff9c4;
            font-style: italic;
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .screenshot-img {
            max-width: 100%;
            border-radius: 6px;
            margin: 6px 0;
            border: 1px solid #bbb;
        }
    </style>
    <!-- Load noVNC core library -->
    <script src="/static/novnc-core.min.js"></script>
</head>
<body>
    <div id="vnc-container">
        <div id="vnc-canvas"></div>
        <div id="vnc-status" style="position: absolute; bottom: 10px; left: 10px; color: white; background: rgba(0,0,0,0.7); padding: 5px; border-radius: 3px;">
            VNC Status: Initializing...
        </div>
    </div>
    <div id="controls-container">
        <h2 class="h4">Chat</h2>
        <div id="chat-output"></div>
        <h2 class="h4 mt-3">Tool Log</h2>
        <div id="tool-log"></div>
        <div id="user-input" class="mt-auto">
            <input type="text" id="message-input" class="form-control" placeholder="Type your message...">
            <button onclick="sendMessage()" class="btn btn-primary">Send</button>
        </div>
        <p class="mt-2">
            <small class="text-muted">
                LM Studio: {{ lmstudio_base }} | Model: {{ model_name }} |
                VNC: 127.0.0.1:{{ rfb_port }} (external)
            </small>
        </p>
    </div>

    <script src="/static/socket.io.js"></script>
    <script>
        // Initialize VNC and Socket.IO
        let rfb = null;
        
        // Set up socket connection
        window.chatSocket = io(window.location.origin, {
            path: "/socket.io/",
            transports: ['websocket', 'polling']
        });
        
        // Initialize VNC connection
        function initVNC() {
            try {
                // Check if RFB is available
                if (typeof RFB !== 'function') {
                    console.error('RFB not available - noVNC library not loaded correctly');
                    document.getElementById('vnc-status').textContent = 'Error: noVNC not loaded';
                    document.getElementById('vnc-status').style.background = 'rgba(255,0,0,0.7)';
                    return;
                }
                
                console.log('Initializing VNC connection');
                const host = window.location.hostname;
                const wsURL = `ws://${host}:12346`;
                
                // Get the VNC canvas
                const vncCanvas = document.getElementById('vnc-canvas');
                
                // Update status
                document.getElementById('vnc-status').textContent = `Connecting to ${wsURL}...`;
                
                // Create RFB connection
                rfb = new RFB(vncCanvas, wsURL, {
                    credentials: { password: null },
                    shared: true,
                    reconnect: true
                });
                
                // Add event listeners
                rfb.addEventListener('connect', () => {
                    console.log('VNC connected successfully');
                    document.getElementById('vnc-status').textContent = 'VNC Status: Connected';
                    document.getElementById('vnc-status').style.background = 'rgba(0,128,0,0.7)';
                });
                
                rfb.addEventListener('disconnect', (e) => {
                    console.log('VNC disconnected', e);
                    document.getElementById('vnc-status').textContent = 'VNC Status: Disconnected - Retrying in 5s';
                    document.getElementById('vnc-status').style.background = 'rgba(255,0,0,0.7)';
                    
                    // Try to reconnect after a delay
                    setTimeout(initVNC, 5000);
                });
                
                rfb.addEventListener('credentialsrequired', () => {
                    console.log('VNC credentials required');
                    document.getElementById('vnc-status').textContent = 'VNC Status: Credentials Required';
                });
            } catch (err) {
                console.error('Error initializing VNC connection:', err);
                document.getElementById('vnc-status').textContent = `VNC Error: ${err.message}`;
                document.getElementById('vnc-status').style.background = 'rgba(255,0,0,0.7)';
                
                // Try again after a delay
                setTimeout(initVNC, 5000);
            }
        }
        
        // Initialize on page load
        window.addEventListener('DOMContentLoaded', () => {
            console.log('DOM content loaded - initializing...');
            // Allow a moment for other scripts to load
            setTimeout(initVNC, 1000);
        });
        
        // Socket.IO event handlers
        chatSocket.on('connect', () => {
            console.log('Socket.IO connected');
        });
        
        chatSocket.on('connect_error', (err) => {
            console.error('Socket.IO connection error:', err);
            chatSocket.emit('client_js_error', {
                message: 'Socket.IO Connection Error',
                error: err.toString()
            });
        });
        
        chatSocket.on('disconnect', (reason) => {
            console.warn('Socket.IO disconnected:', reason);
        });
        
        chatSocket.on('chat_update', function(data) {
            console.log('Received chat update:', data);
            const chatOutput = document.getElementById('chat-output');
            const msgDiv = document.createElement('div');
            msgDiv.classList.add('message');
            
            if (data.role === 'user') {
                msgDiv.classList.add('user-message');
            } else if (data.role === 'bot') {
                msgDiv.classList.add('bot-message');
            } else {
                msgDiv.classList.add('system-message');
            }
            
            msgDiv.innerHTML = `<strong>${data.role}:</strong> <pre>${data.content}</pre>`;
            chatOutput.appendChild(msgDiv);
            chatOutput.scrollTop = chatOutput.scrollHeight;
        });
        
        chatSocket.on('tool_log_update', function(data) {
            console.log('Tool log update:', data);
            const toolLog = document.getElementById('tool-log');
            const entryDiv = document.createElement('div');
            entryDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre><hr>`;
            toolLog.appendChild(entryDiv);
            toolLog.scrollTop = toolLog.scrollHeight;
        });
        
        chatSocket.on('screenshot_update', function(data) {
            console.log('Screenshot update received');
            if (!data.image_b64) return;
            
            const chatOutput = document.getElementById('chat-output');
            const imgElem = document.createElement('img');
            imgElem.src = 'data:image/png;base64,' + data.image_b64;
            imgElem.className = 'screenshot-img';
            chatOutput.appendChild(imgElem);
            chatOutput.scrollTop = chatOutput.scrollHeight;
        });
        
        // Send message function
        function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            if (message === '') return;
            
            chatSocket.emit('user_message', { message: message });
            input.value = '';
        }
        
        // Expose sendMessage function globally for the button
        window.sendMessage = sendMessage;
        
        // Handle Enter key in message input
        document.getElementById('message-input').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

# Ensure _latest_screenshot_b64 is properly initialized
_latest_screenshot_b64 = None
_latest_screenshot_lock = threading.Lock()

async def update_latest_screenshot_b64_async():
    """Update the latest screenshot base64 string asynchronously.

    This function captures a new screenshot and updates the global _latest_screenshot_b64 variable.
    It's designed to be called when a fresh screenshot is needed without blocking.
    """
    global _latest_screenshot_b64
    try:
        res = await take_screenshot()
        if res.get('status') == 'success':
            with _latest_screenshot_lock:
                _latest_screenshot_b64 = res['image_b64']
    except Exception as e:
        logger.error("Error in update_latest_screenshot_b64_async: %s", e)

@app.route('/')
def index():
    """Render the main UI page."""
    return render_template_string(HTML_TEMPLATE,
                                  lmstudio_base=LMSTUDIO_BASE,
                                  model_name=MODEL_NAME,
                                  rfb_port=RFB_PORT)

# This endpoint is temporary for fetching screenshots if needed by UI, will be replaced by VNC
@app.route('/latest_screenshot_b64')
def get_latest_screenshot_b64_route():
    """Return the latest screenshot as a base64-encoded string."""
    global _latest_screenshot_b64, _latest_screenshot_lock
    try:
        # If a screenshot is already cached, return it immediately
        with _latest_screenshot_lock:
            if _latest_screenshot_b64:
                return jsonify({"image_b64": _latest_screenshot_b64})

        # Otherwise, capture a new screenshot synchronously
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(take_screenshot(), loop)
            res = fut.result(timeout=10)
        else:
            res = loop.run_until_complete(take_screenshot())

        if res.get('status') == 'success':
            with _latest_screenshot_lock:
                _latest_screenshot_b64 = res['image_b64']
            return jsonify({"image_b64": res['image_b64']})
        else:
            return jsonify({"error": res.get('message', 'Screenshot failed')}), 500
    except Exception as e:
        logger.error("Error in /latest_screenshot_b64: %s", e)
        return jsonify({"error": str(e)}), 500

# SocketIO event handlers
@sio.on('client_js_error')  # ADDED SERVER-SIDE HANDLER
def handle_client_js_error(data):
    """Handle client-side JavaScript errors."""
    logger.error("Client-side JavaScript Error: %s", data)

@sio.on('connect', namespace='/')  # Explicitly for default namespace
def handle_default_namespace_connect(sid=None, environ=None):
    """Handle connection to default namespace."""
    logger.info("[SERVER] Client connected to DEFAULT NAMESPACE ('/'). SID: %s, Environ: %s",
                sid, environ)

@sio.on('disconnect')
def handle_generic_disconnect(sid):
    """Handle disconnection from default namespace."""
    logger.info("[SERVER] Client disconnected from default namespace. SID: %s", sid)

@sio.on('user_message')
def handle_user_message(data):
    """Handle user messages from the UI."""
    user_msg = data['message']
    logger.info("Received user message via SocketIO: %s", user_msg)

    CHAT_HISTORY.append(("user", user_msg))
    sio.emit('chat_update', {'role': 'user', 'content': user_msg})

    # This function will be the target for crash_on_thread_exception
    def agent_task_sync_wrapper():
        asyncio.run(agent_turn(user_msg))

    crash_on_thread_exception(agent_task_sync_wrapper)

async def chat_llm(messages:list, functions:list):
    """Call LM Studio; returns role / content / function_call dict."""
    resp = await openai.ChatCompletion.acreate(
        model=MODEL_NAME,
        messages=messages,
        functions=functions,
        function_call="auto",
        temperature=0.2,
    )
    return resp.choices[0].message

# ---------- helpers ----------
async def arun(cmd:str, timeout:float=120.0) -> Tuple[int, str, str]:
    """Run a shell command asynchronously and return returncode, stdout, and stderr.

    Args:
        cmd: The shell command to run
        timeout: Maximum time in seconds to wait for the command to complete

    Returns:
        Tuple of (return_code, stdout, stderr)

    Raises:
        asyncio.TimeoutError: If the command does not complete within the timeout
    """
    p = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(p.communicate(), timeout)
    except asyncio.TimeoutError:
        logger.error("Command '%s' timed out after %s seconds.", cmd, timeout)
        p.kill()
        raise
    return p.returncode, out.decode(), err.decode()

async def run(cmd:str, *a, **kw):
    """Alias for arun."""
    return await arun(cmd, *a, **kw)

async def xd_command(*p):
    """Run a command with the DISPLAY environment variable set.

    Args:
        *p: Command arguments to be shell-quoted

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = "DISPLAY=%s %s" % (DISPLAY, ' '.join(map(shlex.quote, p)))
    return await run(cmd)

# ---------- desktop / bash / edit tools ----------
def _tool_wrap(fn):
    """Wrap a tool function to prevent pydantic warning in gradio.

    Args:
        fn: The function to wrap

    Returns:
        The wrapped function
    """
    fn.__signature__ = None  # prevent pydantic warning in gradio
    return fn

@_tool_wrap
async def bash(command:str) -> Dict[str, str]:
    """Run arbitrary shell command and capture stdout/stderr.
    Args:
        command: The shell command to run
    Returns:
        Dict with status, stdout, stderr, and exit code
    """
    logger.info("Running bash command: %s", command)
    code, out, err = await run(command)
    if code != 0:
        logger.warning("Bash command failed with code %s. Stdout: %s, Stderr: %s", code, out, err)
    return {
        "status": "success" if code==0 else "error",
        "stdout": out,
        "stderr": err,
        "code": code
    }

# Minimal filesystem-based editor (view / insert / replace / undo) ----
UNDO_DIR = Path(tempfile.gettempdir()) / "agent_file_undo"
UNDO_DIR.mkdir(exist_ok=True)

@_tool_wrap
def create(path: str, content: str = "") -> str:
    """Create a new file or overwrite an existing one with the specified content.

    Args:
        path: The path to the file to create
        content: The content to write to the file

    Returns:
        "created" if successful, otherwise "error: <message>"
    """
    logger.info("Creating file: %s", path)
    try:
        p = Path(path)
        if p.exists():
            bak = UNDO_DIR / (p.name + ".bak")
            shutil.copy2(p, bak)
        # Atomic write
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(content, encoding='utf-8')
        tmp.replace(p)
        return "created"
    except Exception as e:
        logger.error("Error creating file %s: %s", path, e)
        return "error: %s" % e

@_tool_wrap
def view(path: str) -> str:
    """View the contents of a file.

    Args:
        path: The path to the file to view

    Returns:
        The contents of the file, or "error: <message>" if the file cannot be read
    """
    logger.info("Viewing file: %s", path)
    try:
        p = Path(path)
        if not p.exists():
            return "error: file %s does not exist" % path
        return p.read_text(encoding='utf-8')
    except Exception as e:
        logger.error("Error viewing file %s: %s", path, e)
        return "error: %s" % e

@_tool_wrap
def insert(path: str, line: int, text: str) -> str:
    """Insert text at the specified line number in a file.

    Args:
        path: The path to the file to edit
        line: The line number to insert the text at (0-indexed)
        text: The text to insert

    Returns:
        "inserted" if successful, otherwise "error: <message>"
    """
    logger.info("Inserting into file: %s at line %s", path, line)
    try:
        p = Path(path)
        if not p.exists():
            return "error: file %s does not exist" % path
        bak = UNDO_DIR / (p.name + ".bak")
        shutil.copy2(p, bak)
        lines = p.read_text(encoding='utf-8').splitlines()
        idx = max(0, min(line, len(lines)))
        lines.insert(idx, text)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding='utf-8')
        tmp.replace(p)
        return "inserted"
    except Exception as e:
        logger.error("Error inserting into file %s: %s", path, e)
        return "error: %s" % e

@_tool_wrap
def str_replace(path: str, old: str, new: str) -> str:
    """Replace a string in a file with another string.

    Args:
        path: The path to the file to edit
        old: The string to replace
        new: The string to replace it with

    Returns:
        "replaced" if successful, otherwise "error: <message>"
    """
    logger.info("Replacing in file: %s", path)
    try:
        p = Path(path)
        if not p.exists():
            return "error: file %s does not exist" % path
        bak = UNDO_DIR / (p.name + ".bak")
        shutil.copy2(p, bak)
        content = p.read_text(encoding='utf-8')
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(content.replace(old, new), encoding='utf-8')
        tmp.replace(p)
        return "replaced"
    except Exception as e:
        logger.error("Error replacing in file %s: %s", path, e)
        return "error: %s" % e

@_tool_wrap
def undo_edit() -> str:
    """Undo the last file edit by restoring the most recent backup.

    Returns:
        A message indicating what was undone, "empty" if undo stack is empty,
        or "error: <message>" if there was an error
    """
    logger.info("Undoing last edit.")
    try:
        baks = sorted(UNDO_DIR.glob("*.bak"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not baks:
            logger.warning("Undo stack is empty.")
            return "empty"
        bak = baks[0]
        orig_name = bak.stem
        orig_path = Path(orig_name)
        shutil.copy2(bak, orig_path)
        bak.unlink()
        return "undone: restored %s from backup" % orig_path
    except Exception as e:
        logger.error("Error during undo: %s", e)
        return "error: %s" % e

# -------------------- Tools for Computer Interaction ----------------------
@_tool_wrap
async def take_screenshot() -> Dict[str, str]:
    """Capture full screen PNG and return base64 string and temporary file path.

    Returns:
        Dict with status ('success' or 'error'), and if successful, 'image_b64' (base64 encoded
        image) and 'path' (path to the temporary screenshot file).
    """
    global _latest_screenshot_b64
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    logger.info("Taking screenshot, saving to %s", tmp)
    code, _, err = await xd_command("scrot", "-o", tmp)
    if code:
        logger.error("Failed to take screenshot. Error: %s", err)
        return {"status": "error", "message": err}
    try:
        with open(tmp, "rb") as f:
            img_bytes = f.read()
            b64 = base64.b64encode(img_bytes).decode()
        with _latest_screenshot_lock:
            _latest_screenshot_b64 = b64
        
        # Directly emit the screenshot to the UI
        logger.info("Emitting screenshot to UI from take_screenshot tool")
        sio.emit('screenshot_update', {'image_b64': b64})
        
        return {"status": "success", "image_b64": b64, "path": tmp}
    except Exception as e:
        logger.error("Error encoding screenshot: %s", e)
        return {"status": "error", "message": str(e)}

@_tool_wrap
async def system_output(text: str) -> Dict[str, str]:
    """Send a message to the user as a final response.
    
    Args:
        text: The text to send to the user
        
    Returns:
        Dict with status ('success' or 'error')
    """
    logger.info("System output: %s", text[:100] + ("..." if len(text) > 100 else ""))
    
    # Add to chat history and send to UI
    CHAT_HISTORY.append(("bot", text))
    sio.emit('chat_update', {'role': 'bot', 'content': text})
    
    return {"status": "success", "message": "Response sent to user"}

@_tool_wrap
async def inner_thoughts(text: str) -> Dict[str, str]:
    """Agent's internal thinking process (visible to user as a system message).
    
    Args:
        text: The thinking process to record
        
    Returns:
        Dict with status ('success' or 'error')
    """
    logger.info("Inner thoughts: %s", text[:100] + ("..." if len(text) > 100 else ""))
    
    # Add to chat history and send to UI as a system message
    CHAT_HISTORY.append(("system", f"[Thinking] {text}"))
    sio.emit('chat_update', {'role': 'system', 'content': f"[Thinking] {text}"})
    
    return {"status": "success", "message": "Thinking process recorded"}

async def _xd_click(btn:str, x:int=None, y:int=None):
    """Perform a mouse click at the given coordinates.

    Args:
        btn: The mouse button to click (1=left, 3=right)
        x: Optional x coordinate to move to before clicking
        y: Optional y coordinate to move to before clicking
    """
    if x is not None and y is not None:
        await xd_command("xdotool", "mousemove", str(x), str(y))
    await xd_command("xdotool", "click", btn)

@_tool_wrap
async def computer_action(action: str, **kwargs) -> Dict[str, str]:
    """Perform a computer desktop action and return a screenshot.

    Args:
        action: The type of action from ["mouse_move", "left_click", "right_click",
               "double_click", "key", "type", "scroll"]
        **kwargs: Action-specific parameters:
            - mouse_move: requires x (int), y (int)
            - left_click: optional x (int), y (int)
            - right_click: optional x (int), y (int)
            - double_click: optional x (int), y (int)
            - key: requires key_val (str) (e.g., "Return", "Control_L+c")
            - type: requires text (str)
            - scroll: optional direction (str, "up" or "down"), amount (int)

    Returns:
        Dict with status and screenshot information if successful
    """
    logger.info("Performing computer action: %s with args: %s", action, kwargs)
    try:
        # Validate coordinates
        if action in ("mouse_move", "left_click", "right_click", "double_click"):
            x = kwargs.get("x")
            y = kwargs.get("y")
            if x is not None and (x < 0 or x > 1280):
                raise ValueError("x coordinate %s out of bounds (0-1280)" % x)
            if y is not None and (y < 0 or y > 1024):
                raise ValueError("y coordinate %s out of bounds (0-1024)" % y)
        if action == "key":
            key_val = kwargs.get("key_val")
            if not key_val or not isinstance(key_val, str):
                raise ValueError("key_val must be a non-empty string")
        if action == "type":
            text = kwargs.get("text")
            if not isinstance(text, str):
                raise ValueError("text must be a string")

        for attempt in range(3):
            try:
                if action == "mouse_move":
                    await xd_command("xdotool", "mousemove", str(kwargs["x"]), str(kwargs["y"]))
                elif action == "left_click":
                    await _xd_click("1", kwargs.get("x"), kwargs.get("y"))
                elif action == "right_click":
                    await _xd_click("3", kwargs.get("x"), kwargs.get("y"))
                elif action == "double_click":
                    if kwargs.get("x") is not None and kwargs.get("y") is not None:
                        await xd_command(
                            "xdotool", "mousemove", str(kwargs["x"]), str(kwargs["y"])
                        )
                    await xd_command("xdotool", "click", "--repeat", "2", "1")
                elif action == "key":
                    await xd_command("xdotool", "key", kwargs["key_val"])
                elif action == "type":
                    await xd_command("xdotool", "type", "--delay", "50", kwargs["text"])
                elif action == "scroll":
                    direction = kwargs.get("direction", "down")
                    amount = kwargs.get("amount", 3)
                    btn = "5" if direction == "down" else "4"
                    await xd_command(
                        "xdotool", "click", "--repeat", str(amount), "--delay", "50", btn
                    )
                else:
                    logger.error("Unknown computer_action sub-action: '%s'", action)
                    raise ValueError("Unknown computer_action sub-action: '%s'" % action)
                break
            except Exception as e:
                logger.warning(
                    "Attempt %s failed for computer_action %s: %s",
                    attempt+1, action, e
                )
                if attempt == 2:
                    raise
                await asyncio.sleep(0.2)

        delay = 0.2 if action == "type" else 0.1
        await asyncio.sleep(delay)
        logger.info("Computer action '%s' completed, taking screenshot.", action)
        return await take_screenshot()
    except Exception as e:
        logger.error("Error in computer_action: %s", e)
        return {"status": "error", "message": str(e)}

TOOLS = {
    "bash": bash, "screenshot": take_screenshot,  # screenshot remains for explicit calls
    "computer_action": computer_action,
    "create": create, "view": view, "insert": insert,
    "str_replace": str_replace, "undo_edit": undo_edit,
    "system_output": system_output, "inner_thoughts": inner_thoughts
}

FUNCTION_SCHEMAS = [
    {
        "name": "bash",
        "description": bash.__doc__,
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    },
    {
        "name": "screenshot",
        "description": take_screenshot.__doc__,
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "computer_action",
        "description": computer_action.__doc__,  # Docstring explains conditional args
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["mouse_move", "left_click", "right_click",
                             "double_click", "key", "type", "scroll"]
                },
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "key_val": {"type": "string"},
                "text": {"type": "string"},
                "direction": {"type": "string", "enum": ["up", "down"]},
                "amount": {"type": "integer"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "system_output",
        "description": system_output.__doc__,
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        }
    },
    {
        "name": "inner_thoughts",
        "description": inner_thoughts.__doc__,
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        }
    },
    {
        "name": "create",
        "description": create.__doc__,
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path"]
        }
    },
    {
        "name": "view",
        "description": view.__doc__,
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    },
    {
        "name": "insert",
        "description": insert.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "line": {"type": "integer"},
                "text": {"type": "string"}
            },
            "required": ["path", "line", "text"]
        }
    },
    {
        "name": "str_replace",
        "description": str_replace.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"}
            },
            "required": ["path", "old", "new"]
        }
    },
    {
        "name": "undo_edit",
        "description": undo_edit.__doc__,
        "parameters": {"type": "object", "properties": {}}
    }
]

# ---------- RFB (VNC) headless server ----------
# Global process variables for Xvfb, openbox, x11vnc
xvfb_proc = None
openbox_proc = None
x11vnc_proc = None

def cleanup_processes():
    """Clean up child processes on exit."""
    logger.info("Cleaning up child processes...")
    global xvfb_proc, openbox_proc, x11vnc_proc
    for proc, name in [(x11vnc_proc, "x11vnc"), (openbox_proc, "openbox"), (xvfb_proc, "Xvfb")]:
        if proc and proc.poll() is None:
            logger.info("Terminating %s...", name)
            proc.terminate()
            try:
                proc.wait(timeout=5)
                logger.info("%s terminated.", name)
            except subprocess.TimeoutExpired:
                logger.warning("%s did not terminate in time, killing.", name)
                proc.kill()
                proc.wait()
                logger.info("%s killed.", name)

atexit.register(cleanup_processes)

def start_x_and_vnc():
    """Start X environment and VNC server.

    This function sets up the headless X environment and VNC server.
    It starts Xvfb, openbox, and x11vnc and performs health checks to ensure they're running.
    """
    global xvfb_proc, openbox_proc, x11vnc_proc
    logger.info("Starting X environment and VNC server...")

    # Extract display number from DISPLAY string
    display_num_str = DISPLAY.rsplit(':', maxsplit=1)[-1]
    if not display_num_str.isdigit():
        fatal("Invalid DISPLAY format: %s. Cannot extract display number.", DISPLAY)
    display_num = int(display_num_str)

    # Remove stale lock file
    lock_file = "/tmp/.X%s-lock" % display_num
    if os.path.exists(lock_file):
        try:
            logger.info("Attempting to remove stale Xvfb lock file: %s", lock_file)
            os.remove(lock_file)
            logger.info("Successfully removed lock file: %s", lock_file)
        except Exception as e:
            logger.warning(
                "Could not remove lock file %s: %s. This might cause Xvfb to fail, but proceeding.",
                lock_file, e
            )

    logger.info("Starting Xvfb on display %s...", DISPLAY)
    xvfb_cmd_list = [
        "Xvfb", DISPLAY, "-screen", "0", "1280x1024x24",
        "+extension", "RANDR", "-nolisten", "tcp", "-ac"
    ]

    # Start Xvfb process
    xvfb_proc = subprocess.Popen(
        xvfb_cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(2)
    if xvfb_proc.poll() is not None:
        stdout, stderr = xvfb_proc.communicate()
        fatal(
            "Xvfb failed to start. Exit code: %s\nstdout: %s\nstderr: %s",
            xvfb_proc.poll(), stdout.decode().strip(), stderr.decode().strip()
        )
    logger.info("✔︎ Xvfb started successfully.")

    # Set up child environment
    child_env = os.environ.copy()
    child_env["DISPLAY"] = DISPLAY
    if "WAYLAND_DISPLAY" in child_env:
        del child_env["WAYLAND_DISPLAY"]
    if "XDG_SESSION_TYPE" in child_env and child_env["XDG_SESSION_TYPE"].lower() == "wayland":
        del child_env["XDG_SESSION_TYPE"]

    # Start openbox
    logger.info("Starting Openbox...")
    openbox_cmd_list = ["openbox"]
    openbox_proc = subprocess.Popen(
        openbox_cmd_list, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(1)
    if openbox_proc.poll() is not None:
        stdout, stderr = openbox_proc.communicate()
        fatal(
            "Openbox failed to start. Exit code: %s\nstdout: %s\nstderr: %s",
            openbox_proc.poll(), stdout.decode().strip(), stderr.decode().strip()
        )
    logger.info("✔︎ Openbox started successfully.")

    # Start x11vnc
    logger.info("Starting x11vnc on port %s...", RFB_PORT)
    x11vnc_cmd_list = [
        "x11vnc", "-display", DISPLAY, "-rfbport", str(RFB_PORT),
        "-nopw", "-forever",
        "-rfbauth", "/dev/null",
        "-noxrecord", "-noxfixes", "-noxdamage",
        "-skip_lockkeys", "-defer", "10", "-wait", "20", "-localhost"
    ]
    x11vnc_proc = subprocess.Popen(
        x11vnc_cmd_list, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(5)
    if x11vnc_proc.poll() is not None:
        stdout, stderr = x11vnc_proc.communicate()
        fatal(
            "x11vnc failed to start. Exit code: %s\nstdout: %s\nstderr: %s",
            x11vnc_proc.poll(), stdout.decode().strip(), stderr.decode().strip()
        )
    logger.info("✔︎ x11vnc started successfully.")

    # Health check: RFB handshake
    logger.info("Running RFB handshake check on 127.0.0.1:%s...", RFB_PORT)
    try:
        with socket.create_connection(("127.0.0.1", RFB_PORT), timeout=3) as vnc_socket:
            banner = vnc_socket.recv(12)
            if not banner.startswith(b"RFB"):
                logger.error("Banner check FAILED. Banner: %r", banner)
                fatal("x11vnc did not return a valid RFB banner. Received: %r", banner)
            logger.info(
                "✔︎ x11vnc RFB handshake successful (Banner: %r). VNC server appears to be up.",
                banner
            )
    except Exception as e:
        logger.error("Exception during RFB handshake: %s", e)
        fatal("Error connecting to x11vnc for RFB handshake on 127.0.0.1:%s: %s", RFB_PORT, e)

    logger.info("X environment and VNC server startup process completed.")

    # Take initial screenshot for /latest_screenshot_b64
    try:
        logger.info("Taking initial screenshot after Xvfb/Openbox startup...")
        event_loop = asyncio.get_event_loop()
        if event_loop.is_running():
            event_loop.create_task(take_screenshot())
        else:
            event_loop.run_until_complete(take_screenshot())
        logger.info("Initial screenshot taken and stored in _latest_screenshot_b64.")
    except Exception as e:
        logger.error("Failed to take initial screenshot after X startup: %s", e)

# ─────────────── Runtime Health Checks ──────────────────────
def runtime_health_checks():
    """Run runtime health checks to ensure all services are operational.

    This function performs several checks to ensure that:
    1. The X server is responding to xdpyinfo
    2. Screenshots can be captured with scrot
    3. The UI port is available for binding
    """
    logger.info("Running runtime health checks...")

    # 1. xdpyinfo sanity check
    try:
        env_with_display = os.environ.copy()
        env_with_display["DISPLAY"] = DISPLAY
        subprocess.run(
            ["xdpyinfo"],
            env=env_with_display,
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        logger.info("✔︎ X server on %s responded to xdpyinfo.", DISPLAY)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        fatal("xdpyinfo check failed for display %s: %s", DISPLAY, e)

    # 2. Quick screenshot with scrot
    tmp_screenshot_path = os.path.join(tempfile.gettempdir(), "hc_runtime_scrot.png")
    try:
        env_with_display = os.environ.copy()
        env_with_display["DISPLAY"] = DISPLAY
        scrot_rc = subprocess.run(
            ["scrot", "-o", tmp_screenshot_path],
            env=env_with_display,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ).returncode

        if scrot_rc != 0 or not os.path.exists(tmp_screenshot_path):
            fatal("scrot failed to capture screenshot during runtime check (code: %s).", scrot_rc)

        if os.path.exists(tmp_screenshot_path):
            os.remove(tmp_screenshot_path)  # Clean up screenshot

        logger.info("✔︎ scrot can capture screenshots (runtime check).")
    except FileNotFoundError as e:
        fatal("scrot command not found during runtime check: %s", e)
    except Exception as e:
        fatal("Error during scrot runtime check: %s", e)

    # 3. UI Port Free (Flask/SocketIO will fail if not, but good to check early)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_ui_socket:
            try:
                test_ui_socket.bind(("0.0.0.0", UI_PORT))
                logger.info("✔︎ UI_PORT %s is available for binding.", UI_PORT)
            except OSError:
                # Try a few alternative ports if the default is in use
                alternative_ports = [7861, 7862, 7863, 7864, 7865]
                for alt_port in alternative_ports:
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as alt_socket:
                            alt_socket.bind(("0.0.0.0", alt_port))
                            logger.info("Original UI_PORT %s was in use. Using alternative port %s.", UI_PORT, alt_port)
                            global UI_PORT
                            UI_PORT = alt_port
                            break
                    except OSError:
                        continue
                else:
                    # If we get here, none of the alternative ports worked
                    fatal("All UI ports (%s and alternatives) are already in use or cannot be bound.", UI_PORT)
    except Exception as e:
        fatal("Error checking UI_PORT %s: %s", UI_PORT, e)

    logger.info("Runtime health checks passed.")

# Serve socket.io.js from static directory
@app.route('/socket.io.js')
def socketio_js():
    """Serve socket.io.js from static directory."""
    return app.send_static_file('socket.io.js')

# --- Main Execution Flow ---
if __name__ == '__main__':
    # Pre-script cleanup of Xvfb, openbox, and x11vnc to avoid process conflicts
    try:
        logger.info("Attempting pre-script killall -9 Xvfb, openbox, x11vnc...")
        subprocess.run(["killall", "-9", "Xvfb"], capture_output=True, check=False)
        subprocess.run(["killall", "-9", "openbox"], capture_output=True, check=False)
        pre_kill_result = subprocess.run(
            ["killall", "-9", "x11vnc"],
            capture_output=True, text=True, check=False
        )
        if pre_kill_result.returncode == 0:
            logger.info("Pre-script killall x11vnc found and killed process(es).")
        else:
            logger.info(
                "Pre-script killall x11vnc: no process found or error (expected if none running)."
            )
        time.sleep(0.5)  # Brief pause after potential kill
        
        # Create static directory if it doesn't exist
        static_dir = Path('static')
        static_dir.mkdir(exist_ok=True)
        
        # Download socket.io.js if it doesn't exist
        socket_io_path = static_dir / 'socket.io.js'
        if not socket_io_path.exists():
            logger.info("Downloading socket.io.js to static directory")
            # Use subprocess to run curl
            download_cmd = [
                "curl", "-s", "-o", str(socket_io_path),
                "https://cdn.socket.io/4.5.4/socket.io.min.js"
            ]
            result = subprocess.run(download_cmd, check=True)
            if result.returncode == 0:
                logger.info("Successfully downloaded socket.io.js")
            else:
                logger.error("Failed to download socket.io.js")
                
        # Download noVNC library if it doesn't exist
        novnc_path = static_dir / 'novnc-core.min.js'
        if not novnc_path.exists():
            logger.info("Downloading noVNC library to static directory")
            # Use subprocess to run curl
            download_cmd = [
                "curl", "-s", "-o", str(novnc_path),
                "https://cdn.jsdelivr.net/npm/@novnc/novnc@1.4.0/core/novnc-core.min.js"
            ]
            result = subprocess.run(download_cmd, check=True)
            if result.returncode == 0:
                logger.info("Successfully downloaded noVNC library")
            else:
                logger.error("Failed to download noVNC library")
                
    except Exception as e_prekill:
        logger.error("Error during pre-script setup: %s", e_prekill)

    try:
        pre_flight_checks()
        start_x_and_vnc()
        runtime_health_checks()

        # --- Find an available port for UI server ---
        ui_port = UI_PORT  # Start with the default
        port_available = False
        max_attempts = 10
        
        for port_attempt in range(ui_port, ui_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
                    test_socket.bind(('127.0.0.1', port_attempt))
                    # If we get here, the port is available
                    ui_port = port_attempt
                    port_available = True
                    break
            except OSError:
                logger.warning("Port %s is in use, trying next port...", port_attempt)
                continue
                
        if not port_available:
            fatal("Could not find an available port after %s attempts", max_attempts)
        
        # Write chosen port to file for test scripts
        with open("ui_port.txt", "w", encoding='utf-8') as f:
            f.write(str(ui_port))

        logger.info(
            "🖥️ Desktop Agent starting Flask server on http://127.0.0.1:%s",
            ui_port
        )

        sio.run(
            app, host='127.0.0.1', port=ui_port,
            debug=False, use_reloader=False, allow_unsafe_werkzeug=True
        )
        logger.info("Flask-SocketIO server has shut down.")

    except SystemExit as e:
        logger.warning("SystemExit caught with code %s. Terminating.", e.code)
        raise
    except Exception as e:
        fatal("Unexpected top-level error in __main__: %s", e)
    finally:
        logger.info("Script __main__ block finished or exiting.")

# Function definitions before the __main__ block
async def agent_turn(user_msg):
    """Handle a user message with a full multi-turn conversation loop.
    
    This function implements a robust agent turn that:
    1. Prompts the model with clear instructions to use function calling
    2. Executes tools in sequence automatically
    3. Ensures screenshots are properly sent to the UI
    4. Handles errors gracefully
    
    Args:
        user_msg: The user's message to respond to
    """
    global _latest_screenshot_b64, _latest_screenshot_lock
    
    try:
        # Add user message to chat history
        CHAT_HISTORY.append(("user", user_msg))
        sio.emit('chat_update', {'role': 'user', 'content': user_msg})
        logger.info("Starting agent_turn with user message: %s", user_msg)

        # Prepare system prompt - crucial for ensuring model produces function calls
        system_prompt = """You are Desktop Agent, a helpful AI assistant with tools to control a desktop environment.
You can use tools to interact with the desktop and help users accomplish tasks.
You MUST use tool function_calling format for ANY action that requires desktop interaction.
Your response MUST ALWAYS be a valid function_call object for tools in JSON format.
Do NOT provide explanations before or after the function_call object.
If you need to provide a response to the user, use the 'system_output' tool.
If you need to think or reason, use the 'inner_thoughts' tool so the user can see your reasoning.

Examples of CORRECT tool usage (always as a function_call object):
{"function_call": {"name": "system_output", "arguments": {"text": "I'll help you with that."}}}
{"function_call": {"name": "inner_thoughts", "arguments": {"text": "I need to analyze this..."}}}
{"function_call": {"name": "mouse_move", "arguments": {"x": 100, "y": 200}}}"""

        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Convert conversation history to messages
        for role, content in CHAT_HISTORY:
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "bot":
                messages.append({"role": "assistant", "content": content})
            else:  # System messages are converted to user messages for clarity
                messages.append({"role": "user", "content": f"[SYSTEM MESSAGE]: {content}"})
        
        # Loop for multi-turn conversation
        response_content = None
        max_turns = 5  # Maximum number of automatic turns to take
        turn_count = 0
        
        while turn_count < max_turns:
            turn_count += 1
            logger.info("Agent turn %d/%d", turn_count, max_turns)
            
            # Take screenshot before each model call to ensure fresh context
            screenshot_data = None
            try:
                # Use a lock to prevent concurrent screenshot captures
                async with _latest_screenshot_lock:
                    screenshot_data = await take_screenshot()
                    _latest_screenshot_b64 = screenshot_data
                    
                # Send screenshot to UI
                if screenshot_data:
                    sio.emit('screenshot_update', {'image_b64': screenshot_data})
                    logger.info("Sent screenshot to UI")
            except Exception as e:
                logger.error(f"Error taking screenshot: {e}")
            
            # Call model with function definitions
            try:
                response = await chat_llm(messages, FUNCTION_SCHEMAS)
                reply = response.choices[0].message
                
                # Process the model's response
                if reply.function_call:
                    # Valid function call found
                    function_name = reply.function_call.name
                    try:
                        function_args = json.loads(reply.function_call.arguments)
                    except json.JSONDecodeError:
                        function_args = {}  # Use empty dict if parsing fails
                    
                    logger.info(f"Function call: {function_name} with args: {function_args}")
                    
                    # Log the tool usage
                    sio.emit('tool_log_update', {
                        'tool': function_name,
                        'args': function_args
                    })
                    
                    # Handle the "system_output" tool which ends the conversation
                    if function_name == "system_output":
                        response_content = function_args.get("text", "")
                        CHAT_HISTORY.append(("bot", response_content))
                        sio.emit('chat_update', {'role': 'bot', 'content': response_content})
                        # End multi-turn after system_output
                        break
                    
                    # Execute the function
                    result = await execute_function(function_name, function_args)
                    result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                    
                    # Add function result to messages for next turn
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": function_name,
                            "arguments": json.dumps(function_args)
                        }
                    })
                    messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": result_str
                    })
                    
                    # If this was a screenshot operation, capture the result for UI
                    if function_name == "take_screenshot":
                        try:
                            async with _latest_screenshot_lock:
                                # The result should be base64 encoded image
                                if isinstance(result, dict) and "image_b64" in result:
                                    _latest_screenshot_b64 = result["image_b64"]
                                    # Send updated screenshot to UI
                                    sio.emit('screenshot_update', {'image_b64': _latest_screenshot_b64})
                        except Exception as e:
                            logger.error(f"Error handling screenshot result: {e}")
                else:
                    # No function call - try to extract content
                    response_content = reply.content or ""
                    logger.warning(f"Model did not return function call, content: {response_content}")
                    
                    # Try to extract JSON from response if possible
                    json_match = re.search(r'{.*}', response_content, re.DOTALL)
                    if json_match:
                        try:
                            json_content = json.loads(json_match.group(0))
                            if "function_call" in json_content:
                                function_call = json_content["function_call"]
                                function_name = function_call.get("name")
                                function_args = function_call.get("arguments", {})
                                
                                # Convert to proper function call format and retry
                                logger.info(f"Extracted function call: {function_name}")
                                
                                # Execute the function
                                result = await execute_function(function_name, function_args)
                                result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                                
                                # Add to messages for next turn
                                messages.append({
                                    "role": "assistant",
                                    "content": None,
                                    "function_call": {
                                        "name": function_name,
                                        "arguments": json.dumps(function_args)
                                    }
                                })
                                messages.append({
                                    "role": "function",
                                    "name": function_name,
                                    "content": result_str
                                })
                                
                                # Continue to next turn
                                continue
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.error(f"Failed to parse JSON from response: {e}")
                    
                    # If we couldn't extract a function call, end the turn with an error
                    error_msg = "TOOL_GUARD_VIOLATION: Model did not use proper function calling format."
                    CHAT_HISTORY.append(("system", error_msg))
                    sio.emit('chat_update', {'role': 'system', 'content': error_msg})
                    break
                    
            except Exception as e:
                logger.error(f"Error during model call: {e}")
                error_msg = f"Error during model call: {str(e)}"
                CHAT_HISTORY.append(("system", error_msg))
                sio.emit('chat_update', {'role': 'system', 'content': error_msg})
                break
        
        if turn_count >= max_turns and not response_content:
            response_content = "Maximum turns reached without final response."
            CHAT_HISTORY.append(("system", response_content))
            sio.emit('chat_update', {'role': 'system', 'content': response_content})
            
    except Exception as e:
        logger.error(f"Unexpected error in agent_turn: {e}")
        error_msg = f"Unexpected error: {str(e)}"
        CHAT_HISTORY.append(("system", error_msg))
        sio.emit('chat_update', {'role': 'system', 'content': error_msg})

async def execute_function(function_name, function_args):
    """Execute a function by name with given arguments.
    
    Args:
        function_name: Name of the function to execute
        function_args: Arguments to pass to the function
        
    Returns:
        Result of the function execution
    """
    try:
        # Get the function from the TOOLS dictionary
        tool_fn = TOOLS.get(function_name)
        if not tool_fn:
            logger.error(f"Unknown tool: {function_name}")
            return {"error": f"Unknown tool: {function_name}"}
        
        # Log the attempt
        logger.info(f"Executing tool {function_name} with args: {str(function_args)[:200]}...")
        
        # Execute the function
        if asyncio.iscoroutinefunction(tool_fn):
            result = await tool_fn(**function_args)
        else:
            result = tool_fn(**function_args)
            
        # Process screenshot results specially
        if function_name in ("take_screenshot", "screenshot") and isinstance(result, dict) and "image_b64" in result:
            # Ensure the image_b64 field is present and valid
            if not result.get("image_b64"):
                logger.warning("Screenshot function returned empty image_b64 field")
                result["error"] = "Empty screenshot"
            else:
                logger.info(f"Screenshot captured successfully, {len(result['image_b64'])} bytes")
                # Send to UI via Socket.IO
                sio.emit('screenshot_update', {'image_b64': result["image_b64"]})
                
        # Return the result
        return result
    except Exception as e:
        logger.error(f"Error executing function {function_name}: {e}")
        return {"error": str(e)}

class XvfbDisplayManager:
    """Manage an X virtual framebuffer display with VNC access."""

    def __init__(self, display=99, width=1024, height=768, depth=24, rfb_port=12345):
        """Initialize the display manager.
        
        Args:
            display: X display number
            width: Screen width
            height: Screen height
            depth: Color depth
            rfb_port: Port for VNC connections
        """
        self.display = display
        self.width = width
        self.height = height
        self.depth = depth
        self.rfb_port = rfb_port
        self.display_str = f":{self.display}"
        
        # Set DISPLAY environment variable
        os.environ["DISPLAY"] = self.display_str
        
        # Processes
        self.xvfb_process = None
        self.wm_process = None
        self.vnc_process = None
        self.vnc_proxy = None
        
        self._lock = asyncio.Lock()
        
    async def start(self):
        """Start Xvfb, window manager and VNC."""
        async with self._lock:
            try:
                self.log("Starting Xvfb, window manager and VNC")
                
                # Kill any existing processes
                await self.stop()
                
                # Start Xvfb
                cmd = [
                    "Xvfb", self.display_str, "-screen", "0", 
                    f"{self.width}x{self.height}x{self.depth}", 
                    "-listen", "tcp", "-ac", "+extension", "RANDR"
                ]
                self.log("Launching Xvfb: %s", " ".join(cmd))
                self.xvfb_process = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                
                # Wait a moment for X to start
                await asyncio.sleep(1)
                
                # Check if running
                if self.xvfb_process.returncode is not None:
                    # Try to read error output
                    _, stderr = await self.xvfb_process.communicate()
                    stderr_str = stderr.decode('utf-8', errors='ignore')
                    self.log("Xvfb failed to start: %s", stderr_str)
                    return False
                
                # Start window manager (openbox)
                self.log("Starting window manager (openbox)")
                self.wm_process = await asyncio.create_subprocess_exec(
                    "openbox", "--config-file", "/dev/null", 
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                
                # Wait a moment for WM to start
                await asyncio.sleep(1)
                
                # Start x11vnc
                self.log("Starting x11vnc on port %d", self.rfb_port)
                
                # Create a more robust x11vnc command
                vnc_cmd = [
                    "x11vnc", "-display", self.display_str,
                    "-rfbport", str(self.rfb_port),
                    "-shared", "-forever", "-nopw", "-quiet",
                    "-noxrecord", "-noxfixes", "-noxdamage",
                    "-permitfiletransfer", "no", "-tightfilexfer", "no", 
                    "-rfbauth", "none", "-nocursor"
                ]
                
                self.log("Launching x11vnc: %s", " ".join(vnc_cmd))
                self.vnc_process = await asyncio.create_subprocess_exec(
                    *vnc_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                
                # Wait for VNC to be ready
                await asyncio.sleep(2)
                
                # Check if VNC is running
                if self.vnc_process.returncode is not None:
                    _, stderr = await self.vnc_process.communicate()
                    stderr_str = stderr.decode('utf-8', errors='ignore')
                    self.log("x11vnc failed to start: %s", stderr_str)
                    return False
                
                # Start WebSocket proxy for VNC
                self.log("Starting VNC WebSocket proxy on port 12346")
                self.vnc_proxy = VNCWebSocketProxy("0.0.0.0", 12346, "127.0.0.1", self.rfb_port)
                await self.vnc_proxy.start()
                
                self.log("Display manager startup complete")
                return True
            except Exception as e:
                self.log("Error starting display: %s", e)
                await self.stop()
                return False
    
    async def stop(self):
        """Stop all processes."""
        async with self._lock:
            self.log("Stopping display manager")
            
            # Stop WebSocket proxy
            if self.vnc_proxy:
                try:
                    await self.vnc_proxy.stop()
                except Exception as e:
                    self.log("Error stopping VNC proxy: %s", e)
                self.vnc_proxy = None
            
            # Stop x11vnc
            if self.vnc_process:
                try:
                    self.log("Terminating x11vnc process")
                    self.vnc_process.terminate()
                    await asyncio.wait_for(self.vnc_process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.log("x11vnc did not terminate, killing")
                    self.vnc_process.kill()
                except Exception as e:
                    self.log("Error stopping x11vnc: %s", e)
                self.vnc_process = None
            
            # Stop window manager
            if self.wm_process:
                try:
                    self.log("Terminating window manager")
                    self.wm_process.terminate()
                    await asyncio.wait_for(self.wm_process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.log("Window manager did not terminate, killing")
                    self.wm_process.kill()
                except Exception as e:
                    self.log("Error stopping window manager: %s", e)
                self.wm_process = None
            
            # Stop Xvfb
            if self.xvfb_process:
                try:
                    self.log("Terminating Xvfb")
                    self.xvfb_process.terminate()
                    await asyncio.wait_for(self.xvfb_process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.log("Xvfb did not terminate, killing")
                    self.xvfb_process.kill()
                except Exception as e:
                    self.log("Error stopping Xvfb: %s", e)
                self.xvfb_process = None
            
            self.log("Display manager stopped")
                
    async def restart(self):
        """Restart all processes."""
        await self.stop()
        return await self.start()
        
    async def is_running(self):
        """Check if all components are running."""
        if (self.xvfb_process and self.xvfb_process.returncode is None and
            self.wm_process and self.wm_process.returncode is None and
            self.vnc_process and self.vnc_process.returncode is None):
            return True
        return False

