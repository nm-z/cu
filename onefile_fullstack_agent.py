#!/usr/bin/env python3
# onefile_fullstack_agent.py
# ---------------------------------------------------------------
# • All 16 tools (mouse / key / screenshot / bash / edit etc.)
# • Async chat loop that talks to LM-Studio's OpenAI-style endpoint
# • Headless RFB (VNC) server so you can watch & steer from any viewer
# • Gradio web UI (0-install, pure-Python) for local control + chat log
# ---------------------------------------------------------------

import os, sys, json, asyncio, shlex, subprocess, tempfile, base64, textwrap, time, logging # Added logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Literal, List
import nest_asyncio # Import nest_asyncio
import socket, shutil
import requests # For LMStudio and UI health check
import re # Added for re module
import importlib

try:
    eventlet = importlib.import_module('eventlet')
    eventlet.monkey_patch()
    _EVENTLET_AVAILABLE = True
except ImportError:
    _EVENTLET_AVAILABLE = False

# ─────────────── Global Logger for Script ───────────────────
# Configure this early so all parts of the script can use it.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__) # Main logger for the application

# ─────────────── Fail-Fast Setup ─────────────────────────────
def fatal(msg: str, exc_info=False):
    logger.critical(msg, exc_info=exc_info)
    # cleanup_processes() # Potentially unreliable before os._exit or if called from a hook
    # Consider if a minimal, safe cleanup is possible or if immediate exit is paramount.
    # os._exit(1) # Use os._exit for immediate termination from anywhere, including threads
    sys.exit(1) # Use sys.exit to allow atexit handlers to run, hoping to clean up ports

def on_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.warning("KeyboardInterrupt received. Exiting gracefully...")
        # cleanup_processes() # Attempt graceful cleanup on Ctrl+C
        sys.exit(0) # Normal exit for Ctrl+C
    else:
        # Use logger, not fatal, to avoid recursion if logger itself fails
        logging.getLogger("CRITICAL_ERROR_HOOK").critical(
            "Uncaught exception in main thread:", 
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        os._exit(1) # Hard exit for other uncaught exceptions in main thread

sys.excepthook = on_uncaught_exception

def asyncio_loop_exception_handler(loop, context):
    msg = context.get("exception", context.get("message", "Unknown asyncio error"))
    # Use logger, not fatal, to avoid recursion if logger itself fails
    logging.getLogger("CRITICAL_ASYNC_ERROR_HOOK").critical(
        f"Uncaught exception in asyncio event loop: {msg}", 
        exc_info=context.get("exception")
    )
    # loop.stop() # Stop the loop if possible
    os._exit(1) # Hard exit

# Wrapper for creating threads that will call fatal() on unhandled exceptions
def crash_on_thread_exception(target_func, *args, **kwargs):
    def wrapper():
        try:
            target_func(*args, **kwargs)
        except SystemExit as e: # Allow sys.exit() to propagate if called explicitly
            raise e
        except KeyboardInterrupt:
            logger.warning(f"KeyboardInterrupt in thread {threading.current_thread().name}. Exiting thread.")
            # Let main thread handler do the full exit/cleanup for KeyboardInterrupt
        except Exception as e:
            fatal(f"Unhandled exception in thread {threading.current_thread().name}: {e}", exc_info=True)
    
    thread = threading.Thread(target=wrapper, daemon=True) # Ensure daemon=True
    thread.start()
    return thread

# ───────────────  USER CONFIG  ────────────────
LMSTUDIO_BASE   = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
MODEL_NAME      = os.environ.get("LMSTUDIO_MODEL", "qwen2.5-vl-7b-instruct")
DISPLAY         = os.environ.get("DISPLAY", ":99")          # Xephyr/Xvfb seat
RFB_PORT        = int(os.environ.get("RFB_PORT", "12345")) # VNC
UI_PORT         = int(os.environ.get("UI_PORT", "7860"))   # Gradio
# ───────────────────────────────────────────────

# ---------- minimal OpenAI-compatible client ----------
import openai
openai.api_key  = "dummy"
openai.api_base = LMSTUDIO_BASE

# NEW IMPORTS for Flask web server, SocketIO, and noVNC proxying
from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit, Namespace # Import Namespace
import threading # Ensure threading is imported before crash_on_thread_exception
# Make sure 'python-engineio' and 'python-socketio' are installed for flask_socketio
# Make sure 'Flask' is installed

nest_asyncio.apply() # Apply the patch

# Set asyncio exception handler *after* nest_asyncio.apply() and getting the loop
loop = asyncio.get_event_loop() # Get the event loop that nest_asyncio might have patched
loop.set_exception_handler(asyncio_loop_exception_handler)

# ─────────────── Pre-Flight Checks ───────────────────────────
def pre_flight_checks():
    logger.info("Running pre-flight checks...")
    # 1. Python imports (critical ones for core functionality)
    try:
        import flask
        import flask_socketio
        # openai is already imported
        # nest_asyncio is already imported
        logger.info("✔︎ Critical Python dependencies seem to be imported.")
    except ImportError as e:
        fatal(f"Missing critical Python dependency: {e.name}")

    # 2. Binaries on PATH
    required_binaries = ("Xvfb", "openbox", "x11vnc", "scrot", "xdotool", "ss", "xdpyinfo")
    for prog in required_binaries:
        if shutil.which(prog) is None:
            fatal(f"Required binary not found on PATH: {prog}")
    logger.info(f"✔︎ All required binaries found: {', '.join(required_binaries)}")

    # 3. LM Studio API reachable
    if not LMSTUDIO_BASE.startswith("http"): # Basic check
        fatal(f"LMSTUDIO_BASE URL does not look valid: {LMSTUDIO_BASE}")
    try:
        logger.info(f"Checking LM Studio API at {LMSTUDIO_BASE}/models ...")
        # Use a session for potential keep-alive and header management
        with requests.Session() as req_session:
            r = req_session.get(f"{LMSTUDIO_BASE}/models", timeout=5) # Increased timeout slightly
        r.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        logger.info(f"✔︎ LM Studio API reachable at {LMSTUDIO_BASE} and responded successfully.")
    except requests.exceptions.RequestException as e:
        fatal(f"Cannot reach or get a valid response from LMStudio API at {LMSTUDIO_BASE}: {e}")
    except Exception as e: # Catch other potential errors like JSONDecodeError if we check content
        fatal(f"Error during LMStudio API check at {LMSTUDIO_BASE}: {e}")
    logger.info("Pre-flight checks passed.")

# Force DISPLAY to :99, overriding any external environment variable for script execution
os.environ["DISPLAY"] = ":99"
DISPLAY = ":99" # Also update the global constant

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_agent_key!' # Replace with a real secret in production
# Using async_mode='threading' as our agent/tool calls are async via asyncio,
# but Flask itself is sync. SocketIO can bridge this.
# Consider 'asgi' with a suitable ASGI server (like Uvicorn) if full async stack is desired later.
sio = SocketIO(app, cors_allowed_origins="*")

CHAT_HISTORY=[] # Global chat history

# --- VNC Proxying Setup ---
# We'll use a simple target for x11vnc, assuming it's on localhost
VNC_SERVER_HOST = '127.0.0.1'
VNC_SERVER_PORT = RFB_PORT # From global config

class VNCProxyNamespace(Namespace): # Inherit from the imported Namespace
    def __init__(self, namespace=None):
        super().__init__(namespace)
        self.sockets = {} # Initialize sockets here

    def on_connect(self): # MODIFIED: Changed signature
        sid = request.sid
        environ = request.environ
        logger.info(f"VNC client attempting to connect: SID {sid}, Environ: {environ}")
        try:
            # Create a new TCP connection to the real VNC server for each client
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((VNC_SERVER_HOST, VNC_SERVER_PORT))
            logger.info(f"Successfully connected to x11vnc for SID {sid}. Starting forwarder thread.")
            
            # Store the client socket and start a thread to forward data from VNC to WebSocket
            # self.sockets = getattr(self, 'sockets', {}) # MOVED to __init__
            self.sockets[sid] = client_socket

            # Thread to listen to VNC server and forward to WebSocket client
            def forward_vnc_to_ws():
                logger.info(f"VNC->WS forwarder thread started for SID {sid}")
                try:
                    while True:
                        data = client_socket.recv(4096)
                        if not data:
                            logger.info(f"VNC server TCP connection closed for SID {sid}. Stopping forwarder.")
                            break
                        logger.debug(f"SID {sid}: Received {len(data)} bytes from VNC, forwarding to WS.")
                        sio.emit('vnc_data', data, room=sid, namespace='/vnc')
                except socket.error as e: # More specific exception
                    logger.error(f"Socket error in VNC->WS forwarder for SID {sid}: {e}")
                except Exception as e:
                    logger.error(f"Error in VNC->WS forwarder for SID {sid}: {e}")
                finally:
                    logger.info(f"Closing VNC->WS forwarder for SID {sid}")
                    client_socket.close()
                    if sid in self.sockets:
                        del self.sockets[sid]
                    sio.close_room(sid, namespace='/vnc') # Ensure client is disconnected
            
            threading.Thread(target=forward_vnc_to_ws, daemon=True).start()

        except Exception as e:
            logger.error(f"Failed to connect to VNC server for SID {sid}: {e}")
            sio.disconnect(sid, namespace='/vnc')

    def on_vnc_data(self, data): # MODIFIED: Changed signature
        sid = request.sid
        logger.debug(f"VNC client SID {sid} sent {len(data)} bytes to proxy.")
        # Forward data from WebSocket client to VNC server
        if sid in self.sockets:
            try:
                self.sockets[sid].sendall(data)
            except Exception as e:
                logger.error(f"Error sending data to VNC server for SID {sid}: {e}")
                # Consider closing connection here if send fails repeatedly
        else:
            logger.warning(f"Received VNC data for unknown SID {sid}")

    def on_disconnect(self): # MODIFIED: Changed signature
        sid = request.sid
        logger.info(f"VNC client disconnected: SID {sid}. Cleaning up associated VNC TCP socket.")
        if sid in self.sockets:
            self.sockets[sid].close()
            del self.sockets[sid]

    def on_vnc_client_ready(self, data): # MODIFIED: Changed signature
        sid = request.sid
        logger.info(f"Received vnc_client_ready signal from SID {sid} with data: {data}. VNC UI client has connected to proxy.")
        # vnc_client_ready_event.set() # Set the global event

# Register the VNC namespace
sio.on_namespace(VNCProxyNamespace('/vnc'))

# --- HTML Template for the new UI ---
# Updated to include noVNC client and connect to the /vnc WebSocket namespace
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Desktop Agent</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body { 
            font-family: 'Inter', sans-serif;
            display: grid; 
            grid-template-columns: 2fr 1fr; /* VNC takes 2/3, controls 1/3 */
            height: 100vh; 
            margin: 0;
            padding: 0; /* Removed default body padding if any */
        }
        #vnc-container { 
            flex-grow: 1; /* Will be handled by grid column */
            border-right: 1px solid #ccc; 
            position: relative; 
            background-color: #333;
            overflow: hidden; /* Added from quick wins */
        }
        #vnc-canvas { width: 100%; height: 100%; } /* Placeholder for noVNC */
        #controls-container { 
            width: auto; /* Let grid handle width */
            display: flex; 
            flex-direction: column; 
            padding: 1rem; /* Added from quick wins */
            overflow-y: auto; /* Added from quick wins */
        }
        #chat-output, #tool-log { 
            flex-grow: 1; 
            border: 1px solid #eee; 
            padding: .5rem; /* Added from quick wins */
            margin-bottom: 10px; 
            overflow-y: auto;
            border-radius: .5rem; /* Added from quick wins */
        }
        #tool-log { 
            height: 200px; /* Maintained specific height for tool log */
            flex-grow: 0; /* Don't let tool-log grow as much as chat */
        }
        #user-input { display: flex; margin-top: auto; /* Push to bottom */ }
        #user-input input { flex-grow: 1; padding: 8px; border-radius: .375rem 0 0 .375rem; border: 1px solid #ccc; }
        #user-input button { 
            padding: 8px; 
            background: #4f46e5; /* Added from quick wins */
            color: white; /* Added from quick wins */
            border: none; /* Added from quick wins */
            border-radius: 0 .375rem .375rem 0; /* Adjusted for adjacent input */
            cursor: pointer; /* Added from quick wins */
        }
        #user-input button:hover { background: #4338ca; } /* Added from quick wins */

        .message { margin-bottom: 5px; padding: 3px; border-radius: 3px; }
        .user-message { background-color: #e1f5fe; text-align: right; }
        .bot-message { background-color: #f0f0f0; }
        .system-message { background-color: #fff9c4; font-style: italic; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body>
    <div id="vnc-container" class="bg-dark">
        <!-- noVNC will attach here -->
        <div id="vnc-canvas" style="width: 100%; height: 100%;"></div> 
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
        <p class="mt-2"><small class="text-muted">LM Studio: {{ lmstudio_base }} | Model: {{ model_name }} | VNC: 127.0.0.1:{{ rfb_port }} (external)</small></p>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <!-- Add noVNC library -->
    <script type="module">
        // Add a global error handler for JavaScript errors
        window.addEventListener('error', function(event) {
            console.error('Global JavaScript Error:', event.message, 'URL:', event.filename, 'Line:', event.lineno, 'Column:', event.colno, 'Error object:', event.error);
            // Emit error to server if chatSocket is available
            if (typeof chatSocket !== 'undefined' && chatSocket.connected) { 
               chatSocket.emit('client_js_error', { 
                   message: event.message, 
                   filename: event.filename, 
                   lineno: event.lineno, 
                   colno: event.colno,
                   error_obj_str: event.error ? event.error.toString() : 'N/A'
                });
            }
        });
        window.addEventListener('unhandledrejection', function(event) {
            console.error('Unhandled Promise Rejection:', event.reason);
            if (typeof chatSocket !== 'undefined' && chatSocket.connected) {
                chatSocket.emit('client_js_error', { 
                    message: 'Unhandled Promise Rejection', 
                    reason: event.reason ? (typeof event.reason === 'object' ? JSON.stringify(event.reason) : event.reason.toString()) : 'N/A' 
                });
            }
        });

        // import RFB from 'https://cdnjs.cloudflare.com/ajax/libs/noVNC/1.5.0/core/rfb.js'; // COMMENTED OUT
        // import RFB from 'https://cdn.jsdelivr.net/npm/@novnc/novnc/lib/rfb.js'; // New CDN (JsDelivr) - default import failed
        // import RFB from 'https://cdn.jsdelivr.net/npm/@novnc/novnc@latest/build/novnc.min.js'; // Using jsDelivr CDN with default import - 404
        import RFB from 'https://cdn.jsdelivr.net/npm/@novnc/novnc@latest/core/rfb.js'; // Correct path based on docs

        const host = window.location.hostname;
        const port = window.location.port || (window.location.protocol === 'https:' ? 443 : 80);
        // const path = 'vnc'; // Path for the VNC WebSocket proxy defined in Flask - NO LONGER USED FOR URL

        let rfb;
        const vncCanvas = document.getElementById('vnc-canvas');

        class SocketIOToWebsockAdapter {
            constructor() {
                this.sio_socket = io('/vnc', { transports: ['websocket', 'polling'] });
                this._binaryType = 'arraybuffer'; // RFB typically uses binary. Default, can be set by RFB.

                this._onopen = null;
                this._onclose = null;
                this._onmessage = null;
                this._onerror = null;

                this.sio_socket.on('connect', () => {
                    // console.debug('[VNC ADAPTER] Socket.IO connected to /vnc namespace.');
                    if (this._onopen) {
                        // console.debug('[VNC ADAPTER] Calling _onopen()');
                        this._onopen();
                    }
                });

                this.sio_socket.on('disconnect', (reason) => {
                    // console.warn('[VNC ADAPTER] Socket.IO disconnected from /vnc namespace. Reason:', reason);
                    if (this._onclose) {
                        // console.debug('[VNC ADAPTER] Calling _onclose() due to disconnect.');
                        this._onclose({ code: 1000, reason: `Socket.IO disconnected: ${reason}` }); // Simulate a normal close event
                    }
                });

                this.sio_socket.on('vnc_data', (data) => {
                    // console.debug('[VNC ADAPTER] Received vnc_data from server via Socket.IO:', data);
                    if (this._onmessage) {
                        // Data should be ArrayBuffer. If it's string (e.g. base64), RFB will handle it if wsProtocols included 'base64'
                        // However, our server sends binary directly.
                        this._onmessage({ data: data });
                    }
                });

                this.sio_socket.on('connect_error', (err) => {
                    // console.error('[VNC ADAPTER] Socket.IO connection error to /vnc:', err);
                    if (this._onerror) {
                        // console.debug('[VNC ADAPTER] Calling _onerror() due to connect_error.');
                        this._onerror(err);
                    }
                    if (this._onclose) { // Also trigger onclose as the connection failed
                        // console.debug('[VNC ADAPTER] Calling _onclose() due to connect_error.');
                        this._onclose({ code: 1006, reason: `Socket.IO connection error: ${err.message}` });
                    }
                });
            }

            // Methods and properties RFB will set/use
            set onopen(handler) { this._onopen = handler; }
            get onopen() { return this._onopen; }
            set onmessage(handler) { this._onmessage = handler; }
            get onmessage() { return this._onmessage; }
            set onclose(handler) { this._onclose = handler; }
            get onclose() { return this._onclose; }
            set onerror(handler) { this._onerror = handler; }
            get onerror() { return this._onerror; }

            // binaryType property expected by websock.js
            get binaryType() { return this._binaryType; }
            set binaryType(type) { 
                // RFB might try to set this to 'blob' or 'arraybuffer'
                // Socket.IO client itself doesn't have a direct binaryType property for the whole socket,
                // but individual ArrayBuffer emissions are handled. For RFB, we mostly care about receiving binary.
                // The server-side Python Socket.IO should handle incoming binary ArrayBuffers correctly.
                this._binaryType = type;
                // console.log(`[VNC ADAPTER] binaryType set to: ${type}`);
            }

            // protocol property expected by websock.js
            get protocol() { return ''; } // Typically empty if no sub-protocol negotiated

            send(data) {
                // console.debug('[VNC ADAPTER] Sending vnc_data to server via Socket.IO emit:', data);
                this.sio_socket.emit('vnc_data', data);
            }

            close() {
                console.log('[VNC ADAPTER] Close called, disconnecting Socket.IO from /vnc.');
                this.sio_socket.disconnect();
            }

            get readyState() {
                if (this.sio_socket.connected) return 1; // WebSocket.OPEN
                if (this.sio_socket.connecting) return 0; // WebSocket.CONNECTING
                return 3; // WebSocket.CLOSED
            }

            get bufferedAmount() { return 0; } // Simplification, Socket.IO handles buffering
        }


        function connectVNC() {
            console.log("Attempting to connect to VNC via Socket.IO adapter to /vnc namespace...");
            const vncSocketAdapter = new SocketIOToWebsockAdapter();

            rfb = new RFB(vncCanvas, vncSocketAdapter, {
                // wsProtocols: ['binary', 'base64'], // Not needed when passing an adapter
                credentials: { password: null } // Assuming -nopw for x11vnc
            });

            rfb.addEventListener("connect", () => {
                console.log("noVNC connected to RFB via Socket.IO adapter.");
                vncCanvas.style.cursor = 'default';
                // Emit VNC client ready signal to the /vnc namespace
                // The adapter's socket (this.sio_socket) is already connected to /vnc
                if (vncSocketAdapter.sio_socket.connected) {
                     vncSocketAdapter.sio_socket.emit('vnc_client_ready', {sid: vncSocketAdapter.sio_socket.id});
                     console.log("Emitted vnc_client_ready via adapter's socket.");
                } else {
                    console.warn("VNC adapter's socket not connected when trying to emit vnc_client_ready.");
                }
            });

            rfb.addEventListener("disconnect", (detail) => {
                console.warn("noVNC disconnected (event from RFB library):", detail);
                vncCanvas.style.cursor = 'auto';
            });
            
            rfb.addEventListener("credentialsrequired", () => {
                console.warn("noVNC credentials required, but none provided.");
            });

            rfb.addEventListener("desktopname", (evt) => {
                console.log("noVNC remote desktop name: " + evt.detail.name);
            });
            // No explicit rfb.connect() needed here, as RFB will use the adapter's onopen.
            // console.log(`noVNC RFB instance created (URL was in constructor, auto-connect expected)`);
        }

        // Chat SocketIO connection (remains the same)
        console.log('[CLIENT] Attempting to initialize ChatSocket...'); 
        console.log('[CLIENT] Type of io right before chatSocket initialization:', typeof io);
        window.chatSocket = io(window.location.origin, { 
            path: "/socket.io/",
            transports: ['websocket', 'polling'] // Prioritize WebSocket
        }); 
        console.log('[CLIENT] ChatSocket initialized object:', window.chatSocket);

        window.chatSocket.on('connect', () => {
            console.log('[CLIENT] ChatSocket: Successfully connected to server via default namespace. SID:', window.chatSocket.id);
            console.log('[CLIENT] ChatSocket: Emitting chat_client_ready...');
            window.chatSocket.emit('chat_client_ready'); // Emit ready signal
            console.log('[CLIENT] ChatSocket: chat_client_ready emitted.');
            connectVNC(); // UNCOMMENTED
        });

        window.chatSocket.on('connect_error', (err) => {
            console.error('[CLIENT] ChatSocket: Connection Error!', err);
            if (typeof window.chatSocket !== 'undefined') { // No emit if socket itself is broken
                 window.chatSocket.emit('client_js_error', { message: 'ChatSocket Connection Error', error_obj_str: err ? err.toString() : 'N/A' });
            }
        });

        window.chatSocket.on('disconnect', (reason) => {
            console.warn('[CLIENT] ChatSocket: Disconnected from server.', reason);
            if (typeof window.chatSocket !== 'undefined') { // No emit if socket itself is broken
                window.chatSocket.emit('client_js_error', { message: 'ChatSocket Disconnected', reason_str: reason ? reason.toString() : 'N/A' });
            }
        });

        window.chatSocket.on('reconnect_attempt', (attemptNumber) => {
            console.log('ChatSocket: Attempting to reconnect...', attemptNumber); // New log
        });

        window.chatSocket.on('reconnect_error', (err) => {
            console.error('ChatSocket: Reconnection Error!', err); // New log
        });

        window.chatSocket.on('reconnect_failed', () => {
            console.error('ChatSocket: Reconnection Failed after multiple attempts.'); // New log
        });

        window.chatSocket.on('chat_update', function(data) {
            const chatOutput = document.getElementById('chat-output');
            const msgDiv = document.createElement('div');
            msgDiv.classList.add('message');
            if (data.role === 'user') msgDiv.classList.add('user-message');
            else if (data.role === 'bot') msgDiv.classList.add('bot-message');
            else msgDiv.classList.add('system-message');
            msgDiv.innerHTML = `<strong>${data.role}:</strong> <pre>${data.content}</pre>`;
            chatOutput.appendChild(msgDiv);
            chatOutput.scrollTop = chatOutput.scrollHeight;
        });
        
        window.chatSocket.on('tool_log_update', function(data) {
            const toolLog = document.getElementById('tool-log');
            const entryDiv = document.createElement('div');
            entryDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre><hr>`;
            toolLog.appendChild(entryDiv);
            toolLog.scrollTop = toolLog.scrollHeight;
        });

        function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value;
            if (message.trim() === '') return;
            window.chatSocket.emit('user_message', { message: message });
            input.value = '';
        }
        window.sendMessage = sendMessage;
        
        document.getElementById('message-input').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, 
                                  lmstudio_base=LMSTUDIO_BASE, 
                                  model_name=MODEL_NAME,
                                  rfb_port=RFB_PORT)

# Store last screenshot path or base64 for polling (temporary)
_latest_screenshot_b64: Optional[str] = None
_latest_screenshot_lock = threading.Lock()

async def update_latest_screenshot_b64_async():
    global _latest_screenshot_b64
    try:
        # This function is async, but Flask routes are sync.
        # We'll call this from agent_turn or relevant tool.
        # For now, live_screen_sync_wrapper is sync, so we'd call that.
        # Let's assume take_screenshot() is the primary way to get a b64 image.
        # This part needs careful integration with the async tool calls.
        # For the purpose of this Flask endpoint, we'll rely on take_screenshot()
        # being called by the agent and storing its result.
        # This is a placeholder.
        pass 
    except Exception as e:
        logger.error(f"Error in update_latest_screenshot_b64_async: {e}")


# This endpoint is temporary for fetching screenshots if needed by UI, will be replaced by VNC
@app.route('/latest_screenshot_b64')
def get_latest_screenshot_b64_route():
    with _latest_screenshot_lock:
        if _latest_screenshot_b64:
            return jsonify({"image_b64": _latest_screenshot_b64})
        else:
            # Fallback: try to take a new one, but this is sync and might be slow
            # Better for agent to push updates.
            try:
                # Cannot call async live_screen directly here in a sync Flask route
                # without proper async-to-sync bridge or running Flask with an ASGI server.
                # For simplicity, we'll expect the agent to update _latest_screenshot_b64.
                # img_path = live_screen_sync_wrapper()
                # if img_path:
                #     with open(img_path, "rb") as f:
                #         b64_img = base64.b64encode(f.read()).decode('utf-8')
                #     os.remove(img_path)
                #     return jsonify({"image_b64": b64_img})
                return jsonify({"error": "No screenshot available or agent has not pushed one."}), 404
            except Exception as e:
                logger.error(f"Error in /latest_screenshot_b64 fallback: {e}")
                return jsonify({"error": str(e)}), 500


# SocketIO event handlers
@sio.on('client_js_error') # ADDED SERVER-SIDE HANDLER
def handle_client_js_error(data):
    logger.error(f"Client-side JavaScript Error: {data}")

@sio.on('connect', namespace='/') # Explicitly for default namespace
def handle_default_namespace_connect(sid=None, environ=None):
    logger.info(f"[SERVER] Client connected to DEFAULT NAMESPACE ('/'). SID: {sid}, Environ: {environ}")

# @sio.on('connect') # Keep the original one too, just in case, or remove if redundant
# def handle_generic_connect(sid, environ):
#     logger.info(f"[SERVER] Client connected to default namespace. SID: {sid}, Environ: {environ}")

@sio.on('disconnect')
def handle_generic_disconnect(sid):
    logger.info(f"[SERVER] Client disconnected from default namespace. SID: {sid}")

@sio.on('user_message')
def handle_user_message(data):
    user_msg = data['message']
    logger.info(f"Received user message via SocketIO: {user_msg}")
    
    CHAT_HISTORY.append(("user", user_msg))
    sio.emit('chat_update', {'role': 'user', 'content': user_msg})

    # This function will be the target for crash_on_thread_exception
    def agent_task_sync_wrapper(): 
        asyncio.run(agent_turn(user_msg))

    crash_on_thread_exception(agent_task_sync_wrapper)


async def chat_llm(messages:list, functions:list):
    """Call LM Studio; returns role / content / function_call dict."""
    resp = await openai.ChatCompletion.acreate(
        model      = MODEL_NAME,
        messages   = messages,
        functions  = functions,
        function_call = "auto",
        temperature   = 0.2,
    )
    return resp.choices[0].message

# ---------- helpers ----------
async def arun(cmd:str, timeout:float=120.0)->Tuple[int,str,str]:
    p = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(p.communicate(), timeout)
    except asyncio.TimeoutError:
        logger.error(f"Command '{cmd}' timed out after {timeout} seconds.")
        p.kill(); raise
    return p.returncode, out.decode(), err.decode()

async def run(cmd:str,*a,**kw): return await arun(cmd,*a,**kw)

async def XD(*p):
    return await run("DISPLAY=%s %s" % (DISPLAY," ".join(map(shlex.quote,p))))

# ---------- desktop / bash / edit tools ----------
def _tool_wrap(fn):
    fn.__signature__ = None  # prevent pydantic warning in gradio
    return fn

@_tool_wrap
async def bash(command:str)->Dict[str,str]:
    """Run arbitrary shell command and capture stdout/stderr."""
    logger.info(f"Running bash command: {command}")
    code,out,err = await run(command)
    if code != 0:
        logger.warning(f"Bash command failed with code {code}. Stdout: {out}, Stderr: {err}")
    return {"status":"success" if code==0 else "error",
            "stdout":out,"stderr":err,"code":code}

@_tool_wrap
async def take_screenshot()->Dict[str,str]:
    """Capture full screen PNG, return base64 string + tmp path."""
    global _latest_screenshot_b64
    tmp = tempfile.NamedTemporaryFile(suffix=".png",delete=False).name
    logger.info(f"Taking screenshot, saving to {tmp}")
    code,_,err = await XD("scrot","-o",tmp)
    if code:
        logger.error(f"Failed to take screenshot. Error: {err}")
        return {"status":"error","message":err}
    with open(tmp,"rb") as f:
        img_bytes = f.read()
        b64=base64.b64encode(img_bytes).decode()
    with _latest_screenshot_lock: # Update global screenshot
        _latest_screenshot_b64 = b64
    # Emitting screenshot through socketio if a client is expecting it.
    # sio.emit('screenshot_update', {'image_b64': b64}) # Better to do this after agent_turn if screenshot is part of tool result.
    return {"status":"success","image_b64":b64,"path":tmp}

async def _xd_click(btn:str,x:int=None,y:int=None):
    if x is not None and y is not None: await XD("xdotool","mousemove",str(x),str(y))
    await XD("xdotool","click",btn)

@_tool_wrap
async def computer_action(action: str, **kwargs) -> Dict[str, str]:
    """Performs a computer desktop action and returns a screenshot.
    'action' (str) specifies the type of action from ["mouse_move", "left_click", "right_click", "double_click", "key", "type", "scroll"].
    Keyword arguments depend on the action:
    - mouse_move: requires x (int), y (int)
    - left_click: optional x (int), y (int)
    - right_click: optional x (int), y (int)
    - double_click: optional x (int), y (int)
    - key: requires key_val (str) (e.g., "Return", "Control_L+c")
    - type: requires text (str)
    - scroll: optional direction (str, "up" or "down", defaults to "down"), optional amount (int, defaults to 3)
    All actions result in a screenshot being returned.
    """
    logger.info(f"Performing computer action: {action} with args: {kwargs}")
    if action == "mouse_move":
        await XD("xdotool", "mousemove", str(kwargs["x"]), str(kwargs["y"]))
    elif action == "left_click":
        await _xd_click("1", kwargs.get("x"), kwargs.get("y"))
    elif action == "right_click":
        await _xd_click("3", kwargs.get("x"), kwargs.get("y"))
    elif action == "double_click":
        if kwargs.get("x") is not None and kwargs.get("y") is not None:
             await XD("xdotool", "mousemove", str(kwargs["x"]), str(kwargs["y"]))
        await XD("xdotool", "click", "--repeat", "2", "1")
    elif action == "key":
        await XD("xdotool", "key", kwargs["key_val"])
    elif action == "type":
        await XD("xdotool", "type", "--delay", "50", kwargs["text"])
    elif action == "scroll":
        direction = kwargs.get("direction", "down")
        amount = kwargs.get("amount", 3)
        btn = "5" if direction == "down" else "4"
        await XD("xdotool", "click", "--repeat", str(amount), "--delay", "50", btn)
    else:
        logger.error(f"Unknown computer_action sub-action: '{action}'")
        raise ValueError(f"Unknown computer_action sub-action: '{action}'")

    delay = 0.2 if action == "type" else 0.1
    time.sleep(delay) # Using time.sleep here, consider asyncio.sleep if this causes issues in event loop
    logger.info(f"Computer action '{action}' completed, taking screenshot.")
    return await take_screenshot()

# Minimal in-memory "editor" (view / insert / replace / undo) ----
INMEM_FILES:Dict[str,str]={}
UNDO_STACK:List[Tuple[str,str]]=[]

@_tool_wrap
def create(path:str,content:str="")->str:
    logger.info(f"Creating in-memory file: {path}")
    INMEM_FILES[path]=content; return "created"

@_tool_wrap
def view(path:str)->str:
    logger.info(f"Viewing in-memory file: {path}")
    return INMEM_FILES.get(path,"")

@_tool_wrap
def insert(path:str,line:int,text:str)->str:
    logger.info(f"Inserting into in-memory file: {path} at line {line}")
    body=INMEM_FILES.get(path,"").splitlines()
    body.insert(max(0,min(line,len(body))),text)
    INMEM_FILES[path]="\\n".join(body)
    return "inserted"

@_tool_wrap
def str_replace(path:str,old:str,new:str)->str:
    logger.info(f"Replacing in in-memory file: {path}")
    prev=INMEM_FILES.get(path,"")
    UNDO_STACK.append((path,prev))
    INMEM_FILES[path]=prev.replace(old,new)
    return "replaced"

@_tool_wrap
def undo_edit()->str:
    logger.info("Undoing last edit.")
    if not UNDO_STACK:
        logger.warning("Undo stack is empty.")
        return "empty"
    path,prev = UNDO_STACK.pop()
    INMEM_FILES[path]=prev
    return "undone"

TOOLS = {
    "bash":bash,"screenshot":take_screenshot, # screenshot remains for explicit calls
    "computer_action": computer_action,
    "create":create,"view":view,"insert":insert,"str_replace":str_replace,"undo_edit":undo_edit
}

FUNCTION_SCHEMAS=[{
    "name": "bash",
    "description": bash.__doc__,
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
},{
    "name": "screenshot",
    "description": take_screenshot.__doc__,
    "parameters": {"type": "object", "properties": {}}
},{
    "name": "computer_action",
    "description": computer_action.__doc__, # Docstring explains conditional args
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["mouse_move", "left_click", "right_click", "double_click", "key", "type", "scroll"]},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "key_val": {"type": "string"},
            "text": {"type": "string"},
            "direction": {"type": "string", "enum": ["up", "down"]},
            "amount": {"type": "integer"}
        },
        "required": ["action"]
        # True required fields depend on 'action' value, which is hard to express in basic JSON schema.
        # Model will need to infer from description or we list all as optional above and rely on runtime checks in tool.
        # For now, only action is strictly required for the dispatcher.
    }
},{
    "name": "create",
    "description": create.__doc__,
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path"]}
},{
    "name": "view",
    "description": view.__doc__,
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
},{
    "name": "insert",
    "description": insert.__doc__,
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "line": {"type": "integer"}, "text": {"type": "string"}}, "required": ["path", "line", "text"]}
},{
    "name": "str_replace",
    "description": str_replace.__doc__,
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]}
},{
    "name": "undo_edit",
    "description": undo_edit.__doc__,
    "parameters": {"type": "object", "properties": {}}
}]

# ---------- RFB (VNC) headless server ----------
# (Pure-Python "rfbserver" so we don't depend on x11vnc; keeps it one-file)
# import asyncio, struct, socket, threading # These are already imported or not needed for x11vnc

# Global process variables for Xvfb, openbox, x11vnc
xvfb_proc = None
openbox_proc = None
x11vnc_proc = None # This is now effectively unused for management after launch

def cleanup_processes():
    logger.info("Cleaning up child processes...")
    # x11vnc is cleaned up by killall at the start of start_x_and_vnc 
    # and potentially at script exit if atexit is robust enough against hard crashes.
    # No direct x11vnc_proc to manage here anymore.
            
    if openbox_proc and openbox_proc.poll() is None:
        logger.info("Terminating openbox...")
        openbox_proc.terminate()
        try:
            openbox_proc.wait(timeout=5)
            logger.info("openbox terminated.")
        except subprocess.TimeoutExpired:
            logger.warning("openbox did not terminate in time, killing.")
            openbox_proc.kill()
            openbox_proc.wait()
            logger.info("openbox killed.")

    if xvfb_proc and xvfb_proc.poll() is None:
        logger.info("Terminating Xvfb...")
        xvfb_proc.terminate()
        try:
            xvfb_proc.wait(timeout=5)
            logger.info("Xvfb terminated.")
        except subprocess.TimeoutExpired:
            logger.warning("Xvfb did not terminate in time, killing.")
            xvfb_proc.kill()
            xvfb_proc.wait()
            logger.info("Xvfb killed.")

import atexit # Ensure atexit is imported
atexit.register(cleanup_processes)

# ---------- main agent loop ----------
async def agent_turn(user:str):
    # CHAT_HISTORY.append(("user",user)) # Already done in socketio handler
    # logger.info(f"User input: {user}") # Already logged in socketio handler

    # Construct the detailed system prompt
    datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')
    system_prompt = f"""
You are an AI assistant agent that can control a computer desktop environment.
Your goal is to complete tasks specified by the user.
Your are operating in a Linux environment (Arch Linux) with an Xfce-like desktop managed by Openbox, displayed on a virtual X server (Xvfb) at DISPLAY={DISPLAY}.
You can view the desktop through screenshots provided by tools, and a live VNC feed is available to the user at 127.0.0.1:{RFB_PORT}.
Screen resolution is 1280x1024.

Today is {datetime_str}.

**IMPORTANT RULES:**
1.  **YOU MUST USE A TOOL FOR EVERY ACTION.** Do not try to respond with conversational text unless explicitly asked to reflect or if you are confirming task completion. If you do not call a tool, an error will be raised.
2.  After every action that changes the screen's state (e.g., click, type, scroll), a screenshot will automatically be taken and provided to you as the result of that tool. Use this visual information to guide your next steps.
3.  If you need to use the shell, use the `bash` tool. It provides `stdout`, `stderr`, and `exit_code`. Large outputs will be truncated, so be mindful of commands that produce excessive output.
4.  For file editing, use the provided tools: `create`, `view`, `insert`, `str_replace`, `undo_edit`. These operate on in-memory files.
5.  Be precise with coordinates for mouse actions. The top-left of the screen is (0,0).
6.  When typing, use the `type` tool. For special keys (Enter, Tab, Ctrl+C, etc.), use the `key` tool (e.g., `key` with `key_val="Return"` for Enter, `key` with `key_val="Control_L+c"` for Ctrl+C). Refer to xdotool key names if unsure.
7.  For web browsing, assume a generic browser might be open or can be opened via `bash`. Be aware of common browser elements. If a Firefox first-run wizard appears, try to dismiss it (e.g. with Escape key or clicking a "no thanks" button).
8.  If dealing with PDF files, try to find a text-based way to extract information (e.g., `bash` with `pdftotext your_document.pdf -` to output to stdout) rather than relying solely on scrolling and screenshotting, as this can be inefficient.
9.  Think step-by-step. If a task is complex, break it down.
10. If a tool fails or the outcome is not what you expected, reassess the situation using the latest screenshot and try a different approach or tool.

Available tools are:
- bash: Run arbitrary shell command and capture stdout/stderr.
- screenshot: Capture full screen PNG, return base64 string + tmp path. (This is now automatic after most actions)
- computer_action: Performs a computer desktop action and returns a screenshot.
- create: Creates a new in-memory file with given content.
- view: Views the content of an in-memory file.
- insert: Inserts text at a specific line in an in-memory file.
- str_replace: Replaces text in an in-memory file.
- undo_edit: Undoes the last in-memory file modification.

Always strive to complete the user's request efficiently.
"""
    msgs=[{"role":"system","content":system_prompt}]
    # Create a local copy for this turn to avoid issues if global CHAT_HISTORY is modified elsewhere
    current_chat_history = list(CHAT_HISTORY) 
    for r,c in current_chat_history:
        role="assistant" if r=="bot" else r
        msgs.append({"role":role,"content":c})
    
    logger.info("Calling LLM...")
    try:
        reply=await chat_llm(msgs,FUNCTION_SCHEMAS)
    except Exception as e:
        logger.error(f"Error calling LLM: {e}", exc_info=True)
        sio.emit('chat_update', {'role': 'system', 'content': f"LLM call failed: {str(e)}"})
        CHAT_HISTORY.append(("system", f"LLM_ERROR: {str(e)}"))
        return

    logger.info(f"LLM reply: {reply}")
    sio.emit('chat_update', {'role': 'system', 'content': f"LLM raw reply: {textwrap.shorten(str(reply), 300)}"})


    if not reply.get("function_call"):
        error_message = "LLM did not return a function call. Content: " + reply.get("content", "[no content from LLM]")
        CHAT_HISTORY.append(("system", f"TOOL_GUARD_VIOLATION: {error_message}"))
        logger.error(f"TOOL_GUARD_VIOLATION: {error_message}") 
        sio.emit('chat_update', {'role': 'system', 'content': f"TOOL_GUARD_VIOLATION: {error_message}"})
        # Gradio used to raise ValueError here. For SocketIO, we just inform the client.
        return # End turn

    fn=reply["function_call"]
    name,args_str = fn["name"],fn.get("arguments") 
    logger.info(f"LLM wants to call tool: {name} with args: {args_str}")
    sio.emit('tool_log_update', {"event": "tool_call_attempt", "tool_name": name, "arguments_str": args_str})

    try:
        args=json.loads(args_str or "{}")
    except json.JSONDecodeError as e:
        error_message = f"Failed to decode JSON arguments for tool {name}: {args_str}. Error: {e}"
        CHAT_HISTORY.append(("system", f"JSON_DECODE_ERROR: {error_message}"))
        logger.error(f"JSON_DECODE_ERROR: {error_message}")
        sio.emit('chat_update', {'role': 'system', 'content': f"JSON_DECODE_ERROR: {error_message}"})
        sio.emit('tool_log_update', {"event": "tool_call_error", "tool_name": name, "error": error_message})
        return

    try:
        res=await TOOLS[name](**args) # Await the tool call
        out=textwrap.shorten(str(res), 500) # Increased shorten length for better log
        logger.info(f"Tool {name} result: {out}")
        CHAT_HISTORY.append(("bot",f"[{name} → {out}]"))
        sio.emit('chat_update', {'role': 'bot', 'content': f"Executed: {name}({textwrap.shorten(str(args),100)}).\\nResult: {out}"})
        sio.emit('tool_log_update', {"event": "tool_call_success", "tool_name": name, "arguments": args, "result_summary": out, "full_result": res if len(str(res)) < 2000 else "Result too long for log"})

        # If the tool was a computer_action or screenshot, it would have updated _latest_screenshot_b64
        # and potentially emitted a 'screenshot_update'
        # For other tools, if we decide they need to trigger a general screen refresh for VNC (less relevant with live VNC)
        # or for a fallback screenshot mechanism:
        if name not in ["computer_action", "screenshot"]: # Example: if bash or edit tools might change screen indirectly
            # new_ss = await take_screenshot() # This will update _latest_screenshot_b64 and emit
            # sio.emit('chat_update', {'role': 'system', 'content': f"Took refresh screenshot after {name}: {new_ss.get('status')}"})
            pass # With live VNC, this explicit refresh after other tools is less critical.

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        error_message = f"Error executing tool {name}: {str(e)}"
        CHAT_HISTORY.append(("system", f"TOOL_EXECUTION_ERROR: {error_message}"))
        sio.emit('chat_update', {'role': 'system', 'content': error_message})
        sio.emit('tool_log_update', {"event": "tool_call_error", "tool_name": name, "arguments": args, "error": str(e)})


# ─────────────── X Server and VNC Startup ──────────────────
def start_x_and_vnc():
    global xvfb_proc, openbox_proc, x11vnc_proc
    logger.info("Starting X environment and VNC server...")
    display_num_str = DISPLAY.split(':')[-1]
    if not display_num_str.isdigit():
        fatal(f"Invalid DISPLAY format: {DISPLAY}. Cannot extract display number.")
    display_num = int(display_num_str)

    # --- Cleanup existing processes (without sudo) ---
    cleanup_commands = [
        (["killall", "-9", "x11vnc"], "x11vnc"),
        (["killall", "-9", "Xvfb"], "Xvfb"),
        (["pkill", "-9", "-f", "openbox"], "openbox")
    ]
    for cmd, name in cleanup_commands:
        logger.info(f"Attempting to kill all {name} processes with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Kill command for {name} succeeded.")
        else:
            logger.info(f"Kill command for {name} failed or no process found (code {result.returncode}). Stderr: {result.stderr.strip()}")
        time.sleep(0.2)

    lock_file = f"/tmp/.X{display_num}-lock"
    if os.path.exists(lock_file):
        try:
            logger.info(f"Attempting to remove stale Xvfb lock file: {lock_file}")
            # subprocess.run(["sudo", "-n", "rm", "-f", lock_file], check=True) # With sudo
            subprocess.run(["rm", "-f", lock_file], check=True) # Without sudo
            logger.info(f"Successfully removed lock file: {lock_file}")
        except (OSError, subprocess.CalledProcessError) as e:
            logger.warning(f"Could not remove lock file {lock_file}: {e}. This might cause Xvfb to fail, but proceeding.")
    time.sleep(0.1)

    logger.info(f"Starting Xvfb on display {DISPLAY}...")
    xvfb_cmd_list = ["Xvfb", DISPLAY, "-screen", "0", "1280x1024x24", "+extension", "RANDR", "-nolisten", "tcp", "-ac"]
    xvfb_proc = subprocess.Popen(xvfb_cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2) 
    if xvfb_proc.poll() is not None:
        stdout, stderr = xvfb_proc.communicate()
        fatal(f"Xvfb failed to start. Exit code: {xvfb_proc.poll()}\nstdout: {stdout.decode().strip()}\nstderr: {stderr.decode().strip()}")
    logger.info("✔︎ Xvfb started successfully.")

    child_env = os.environ.copy()
    child_env["DISPLAY"] = DISPLAY
    if "WAYLAND_DISPLAY" in child_env: del child_env["WAYLAND_DISPLAY"]
    if "XDG_SESSION_TYPE" in child_env and child_env["XDG_SESSION_TYPE"].lower() == "wayland":
        del child_env["XDG_SESSION_TYPE"]

    logger.info("Starting Openbox...")
    openbox_cmd_list = ["openbox"]
    openbox_proc = subprocess.Popen(openbox_cmd_list, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1)
    if openbox_proc.poll() is not None:
        stdout, stderr = openbox_proc.communicate()
        fatal(f"Openbox failed to start. Exit code: {openbox_proc.poll()}\nstdout: {stdout.decode().strip()}\nstderr: {stderr.decode().strip()}")
    logger.info("✔︎ Openbox started successfully.")

    logger.info(f"Starting x11vnc on port {RFB_PORT}...")
    x11vnc_cmd_parts = [
        "x11vnc", "-display", DISPLAY, "-rfbport", str(RFB_PORT),
        "-nopw", "-forever", "-bg",
        "-rfbauth", "/dev/null",
        "-noxrecord", "-noxfixes", "-noxdamage",
        "-skip_lockkeys", "-defer", "10", "-wait", "20", "-localhost"
    ]
    # Construct the command string for detached execution
    # Output of x11vnc will go to /dev/null
    nohup_cmd = "nohup " + " ".join(shlex.quote(part) for part in x11vnc_cmd_parts) + " > /dev/null 2>&1 &"

    try:
        logger.info(f"[DEBUG] Attempting to launch x11vnc detached: {nohup_cmd}")
        # We use child_env to ensure DISPLAY is set correctly, shell=True for nohup and &
        subprocess.run(nohup_cmd, shell=True, check=False, env=child_env)
        logger.info("[DEBUG] x11vnc launch command executed.")
        logger.info("[DEBUG] Sleeping for 5.0s to allow x11vnc to start...") # Increased sleep
        time.sleep(5.0) # Increased sleep
        logger.info("[DEBUG] Woke up after sleep for x11vnc.")
    except Exception as e_launch:
        logger.error(f"[DEBUG] Exception during x11vnc detached launch or sleep: {e_launch}", exc_info=True)
        fatal(f"Failed during x11vnc detached launch or sleep: {e_launch}")

    # Since x11vnc_proc is no longer used for polling here, the original poll block is removed.
    # We proceed directly to RFB handshake check.

    logger.info(f"[DEBUG] Just before RFB handshake block on 127.0.0.1:{RFB_PORT}...")
    try:
        logger.info(f"[DEBUG] Attempting socket.create_connection to 127.0.0.1:{RFB_PORT}...")
        vnc_socket = socket.create_connection(("127.0.0.1", RFB_PORT), timeout=3)
        logger.info(f"[DEBUG] socket.create_connection successful. Attempting vnc_socket.recv(12)...")
        banner = vnc_socket.recv(12) 
        logger.info(f"[DEBUG] vnc_socket.recv(12) successful. Banner: {banner!r}. Attempting vnc_socket.close()...")
        vnc_socket.close()
        logger.info(f"[DEBUG] vnc_socket.close() successful. Checking banner content...")
        if not banner.startswith(b"RFB"):
            logger.error(f"[DEBUG] Banner check FAILED. Banner: {banner!r}")
            fatal(f"x11vnc did not return a valid RFB banner. Received: {banner!r}")
        logger.info(f"✔︎ x11vnc RFB handshake successful (Banner: {banner!r}). VNC server appears to be up.")
    except socket.timeout as e_timeout: 
        logger.error(f"[DEBUG] socket.timeout during RFB handshake: {e_timeout}", exc_info=True)
        fatal(f"x11vnc did not respond on 127.0.0.1:{RFB_PORT} within timeout: {e_timeout}")
    except ConnectionRefusedError as e_connrefused: 
        logger.error(f"[DEBUG] ConnectionRefusedError during RFB handshake: {e_connrefused}", exc_info=True)
        fatal(f"x11vnc refused connection on 127.0.0.1:{RFB_PORT}: {e_connrefused}")
    except OSError as e_os: 
        logger.error(f"[DEBUG] OSError during RFB handshake: {e_os}", exc_info=True)
        fatal(f"OSError during RFB handshake on 127.0.0.1:{RFB_PORT}: {e_os}")
    except Exception as e_generic: 
        logger.error(f"[DEBUG] Generic Exception during RFB handshake: {e_generic}", exc_info=True)
        fatal(f"Error connecting to x11vnc for RFB handshake on 127.0.0.1:{RFB_PORT}: {e_generic}")
    
    logger.info("[DEBUG] RFB handshake block completed. Proceeding to 'X environment and VNC server startup process completed.' log")
    logger.info("X environment and VNC server startup process completed.")
    logger.info("[DEBUG] Just after 'X environment and VNC server startup process completed.' log")

# ─────────────── Runtime Health Checks ──────────────────────
def runtime_health_checks():
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
        logger.info(f"✔︎ X server on {DISPLAY} responded to xdpyinfo.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        fatal(f"xdpyinfo check failed for display {DISPLAY}: {e}")

    # 2. Quick screenshot with scrot
    tmp_screenshot_path = os.path.join(tempfile.gettempdir(), "hc_runtime_scrot.png")
    try:
        env_with_display = os.environ.copy()
        env_with_display["DISPLAY"] = DISPLAY
        scrot_rc = subprocess.run(
            ["scrot", "-o", tmp_screenshot_path],
            env=env_with_display,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        if scrot_rc != 0 or not os.path.exists(tmp_screenshot_path):
            fatal(f"scrot failed to capture screenshot during runtime check (code: {scrot_rc}).")
        if os.path.exists(tmp_screenshot_path): os.remove(tmp_screenshot_path) # Clean up screenshot
        logger.info("✔︎ scrot can capture screenshots (runtime check).")
    except FileNotFoundError as e: 
         fatal(f"scrot command not found during runtime check: {e}")
    except Exception as e: 
         fatal(f"Error during scrot runtime check: {e}")

    # 3. UI Port Free (Flask/SocketIO will fail if not, but good to check early)
    try:
        test_ui_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_ui_socket.bind(("0.0.0.0", UI_PORT))
        test_ui_socket.close()
        logger.info(f"✔︎ UI_PORT {UI_PORT} is available for binding.")
    except OSError:
        fatal(f"UI_PORT {UI_PORT} is already in use or cannot be bound.")
    logger.info("Runtime health checks passed.")

# --- Main Execution Flow ---
if __name__ == '__main__':
    # AGGRESSIVE PRE-SCRIPT CLEANUP of x11vnc
    try:
        logger.info("Attempting pre-script killall -9 x11vnc...")
        pre_kill_result = subprocess.run(["killall", "-9", "x11vnc"], capture_output=True, text=True, check=False)
        if pre_kill_result.returncode == 0:
            logger.info("Pre-script killall x11vnc found and killed process(es).")
        else:
            logger.info("Pre-script killall x11vnc: no process found or error (expected if none running).")
        time.sleep(0.5) # Brief pause after potential kill
    except Exception as e_prekill:
        logger.error(f"Error during pre-script x11vnc killall: {e_prekill}")
    try:
        pre_flight_checks() # RE-ENABLED
        start_x_and_vnc()   # RE-ENABLED
        runtime_health_checks() # RE-ENABLED
        logger.info(f"🖥️ Desktop Agent starting Flask server on http://0.0.0.0:{UI_PORT} (minimal startup)")
        if _EVENTLET_AVAILABLE:
            logger.info("[SERVER] Using eventlet WSGI server for Flask-SocketIO.")
            sio.run(app, host='0.0.0.0', port=UI_PORT, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
        else:
            sio.run(app, host='0.0.0.0', port=UI_PORT, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
        logger.info("Flask-SocketIO server has shut down.")
    except SystemExit as e:
        logger.warning(f"SystemExit caught with code {e.code}. Terminating.") 
        if e.code != 0:
             pass 
        raise 
    except Exception as e:
        fatal(f"Unexpected top-level error in __main__: {e}", exc_info=True)
    finally:
        logger.info("Script __main__ block finished or exiting.")
