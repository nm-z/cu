// debug-chat-socket.js
const puppeteer = require('puppeteer');

(async () => {
  // Launch in headed mode so you can see it; switch to headless:true if you prefer
  const browser = await puppeteer.launch({ headless: false, defaultViewport: null });
  const page    = await browser.newPage();

  // 1) Mirror browser console into Node
  page.on('console', msg => {
    console.log(`BROWSER ${msg.type().toUpperCase()}: ${msg.text()}`);
  });

  // 2) Hook into CDP to watch WebSocket events
  const client = await page.target().createCDPSession();
  await client.send('Network.enable');

  client.on('Network.webSocketCreated', ({ requestId, url }) => {
    console.log(`WS CREATED → ${url}`);
  });
  client.on('Network.webSocketFrameSent',    ({ response }) => {
    console.log(`WS FRAME SENT     → ${response.payloadData}`);
  });
  client.on('Network.webSocketFrameReceived',({ response }) => {
    console.log(`WS FRAME RECEIVED → ${response.payloadData}`);
  });
  client.on('Network.webSocketFrameError',   ({ requestId, errorMessage }) => {
    console.error(`WS FRAME ERROR    → ${errorMessage}`);
  });
  client.on('Network.webSocketClosed',       ({ requestId }) => {
    console.log(`WS CLOSED (id=${requestId})`);
  });

  // 3) Navigate to your UI
  await page.goto('http://127.0.0.1:7860', { waitUntil: 'networkidle2' });

  // 4) From inside the page, wait for your Socket.IO client to connect (or error)
  const result = await page.evaluate(() => {
    return new Promise(resolve => {
      if (!window.chatSocket) {
        console.error('[PUPPETEER_EVAL] chatSocket not found on window immediately.');
        return resolve({ status: 'error', reason: 'chatSocket_not_found_immediately' });
      }

      console.log('[PUPPETEER_EVAL] chatSocket found. ID:', window.chatSocket.id, 'Connected:', window.chatSocket.connected);

      const onConnect = () => {
        console.log('[PUPPETEER_EVAL] chatSocket "connect" event fired. ID:', window.chatSocket.id);
        resolve({ status: 'connected', id: window.chatSocket.id });
        clearTimeout(timeoutId);
        window.chatSocket.off('connect', onConnect);
        window.chatSocket.off('connect_error', onConnectError);
      };

      const onConnectError = (err) => {
        console.error('[PUPPETEER_EVAL] chatSocket "connect_error" event fired:', err);
        resolve({ status: 'connect_error', error: err ? JSON.stringify(err) : 'Unknown error' });
        clearTimeout(timeoutId);
        window.chatSocket.off('connect', onConnect);
        window.chatSocket.off('connect_error', onConnectError);
      };

      window.chatSocket.on('connect', onConnect);
      window.chatSocket.on('connect_error', onConnectError);

      // Log status after a short delay if not connected yet
      setTimeout(() => {
        if (!window.chatSocket.connected) {
            console.log('[PUPPETEER_EVAL] chatSocket status after 2s delay: ID:', window.chatSocket.id, 'Connected:', window.chatSocket.connected, 'Disconnected:', window.chatSocket.disconnected);
        }
      }, 2000);


      const timeoutId = setTimeout(() => {
        console.warn('[PUPPETEER_EVAL] Timeout waiting for chatSocket connect/connect_error event.');
        resolve({ status: 'timeout', connected: window.chatSocket.connected, id: window.chatSocket.id });
        window.chatSocket.off('connect', onConnect);
        window.chatSocket.off('connect_error', onConnectError);
      }, 7000); // Increased timeout slightly to 7s
    });
  });

  console.log('▶️ chatSocket connection result:', result);

  // Keep the browser open long enough to see everything
  // await page.waitForTimeout(10000); // Old way
  await new Promise(resolve => setTimeout(resolve, 10000)); // New way
  await browser.close();
})(); 