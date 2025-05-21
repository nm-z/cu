import puppeteer from 'puppeteer';
import fetch from 'node-fetch';
import fs from 'fs';

const UI_URL = process.env.UI_URL || 'http://127.0.0.1:7860/';
const HEADLESS = (process.env.HEADLESS ?? 'true').toLowerCase() !== 'false';

// Increase timeouts for slower systems
const PAGE_LOAD_TIMEOUT = 15000;
const ELEMENT_WAIT_TIMEOUT = 12000; 
const TOOL_WAIT_TIMEOUT = 60000; // 60 seconds to wait for tool execution

// Log function
function log(message) {
  console.log(`[${new Date().toISOString()}] ${message}`);
  fs.appendFileSync('test_log.txt', `[${new Date().toISOString()}] ${message}\n`);
}

(async () => {
  let browser, page;
  const errors = [];
  try {
    log("Starting browser...");
    browser = await puppeteer.launch({ 
      headless: HEADLESS, 
      defaultViewport: null, 
      args: ['--disable-web-security']
    });
    page = await browser.newPage();
    
    // Set up console logging
    page.on('console', msg => log(`BROWSER CONSOLE: ${msg.type()}: ${msg.text()}`));
    page.on('pageerror', err => log(`BROWSER PAGE ERROR: ${err.toString()}`));
    
    // Set up websocket event logging
    page.on('request', request => {
      if (request.url().includes('socket.io') || request.url().includes('ws:')) {
        log(`WS REQUEST: ${request.url()}`);
      }
    });

    log(`Navigating to ${UI_URL}...`);
    await page.goto(UI_URL, { waitUntil: 'domcontentloaded', timeout: PAGE_LOAD_TIMEOUT });
    
    log("Waiting for input field...");
    await page.waitForSelector('#message-input', { timeout: ELEMENT_WAIT_TIMEOUT });
    log("Input field found");

    // 1. VNC Desktop embedded check
    try {
      log("Checking for VNC canvas...");
      await page.waitForSelector('#vnc-canvas', { visible: true, timeout: ELEMENT_WAIT_TIMEOUT });
      
      // Wait a bit to give noVNC time to initialize
      await new Promise(r => setTimeout(r, 3000));
      
      const vncCanvasInfo = await page.evaluate(() => {
        const vncElem = document.getElementById('vnc-canvas');
        return {
          hasChildren: vncElem && vncElem.children.length > 0,
          hasCanvas: vncElem && !!vncElem.querySelector('canvas'),
          childrenCount: vncElem ? vncElem.children.length : 0,
          html: vncElem ? vncElem.innerHTML : 'none'
        };
      });
      
      log(`VNC canvas info: ${JSON.stringify(vncCanvasInfo)}`);
      
      if (!vncCanvasInfo.hasChildren && !vncCanvasInfo.hasCanvas) {
        throw new Error(`VNC canvas is visible but has no child elements (noVNC not embedded). Details: ${JSON.stringify(vncCanvasInfo)}`);
      }
    } catch (e) {
      log(`VNC desktop check failed: ${e.message}`);
      errors.push('FAIL: VNC Desktop is not embedded in the window: ' + e.message);
    }

    // 2. VNC backend running check (try handshake via /latest_screenshot_b64)
    try {
      log("Checking for VNC backend via screenshot endpoint...");
      const resp = await fetch(UI_URL + 'latest_screenshot_b64');
      const data = await resp.json();
      log(`Screenshot API response status: ${resp.status}, has image data: ${!!data.image_b64}`);
      if (!data.image_b64 || data.image_b64.length < 100) {
        throw new Error('No valid screenshot returned');
      }
    } catch (e) {
      log(`VNC backend check failed: ${e.message}`);
      errors.push('FAIL: VNC backend is not running or screenshot not available: ' + e.message);
    }

    // 3. Send a prompt to trigger tool use and multi-turn
    const testPrompt = 'Try each of your tools, one at a time. Do not simulate tool calls. After each, continue to the next. Stop only when all tools are used.';
    log(`Typing test prompt: ${testPrompt}`);
    await page.type('#message-input', testPrompt);
    log("Sending message...");
    await page.keyboard.press('Enter');
    
    log("Waiting for response...");
    await new Promise(r => setTimeout(r, 3000));
    
    // Save page HTML for debugging
    const pageHtml = await page.content();
    fs.writeFileSync('page_debug.html', pageHtml);
    log("Saved page HTML to page_debug.html");

    // 4. Tool use and multi-turn check (wait for at least 2 tool calls in tool log)
    let toolLogEntries = [];
    log("Waiting for tool log entries...");
    
    for (let i = 0; i < 20; ++i) {
      toolLogEntries = await page.evaluate(() => {
        return Array.from(document.querySelectorAll('#tool-log pre')).map(e => e.innerText);
      });
      
      log(`Tool log entries (${i}): ${toolLogEntries.length}`);
      
      if (toolLogEntries.length >= 2) {
        log("Found multiple tool log entries!");
        break;
      }
      
      // Log contents of first entry if available
      if (toolLogEntries.length > 0) {
        log(`First tool log entry: ${toolLogEntries[0].substring(0, 200)}...`);
      }
      
      await new Promise(r => setTimeout(r, 3000));
    }
    
    // Save tool log entries
    fs.writeFileSync('tool_log_entries.json', JSON.stringify(toolLogEntries, null, 2));
    
    if (toolLogEntries.length < 2) {
      log(`FAIL: Not enough tool log entries: ${toolLogEntries.length}`);
      errors.push('FAIL: Model did not take more than 1 turn automatically (tool log entries: ' + toolLogEntries.length + ')');
    }

    // 5. Tool call correctness (must not just describe, must execute)
    log("Checking for tool call success...");
    const toolCallOk = toolLogEntries.some(e => /tool_call_success/.test(e));
    if (!toolCallOk) {
      log("FAIL: No tool_call_success found in log");
      errors.push('FAIL: Model did not use tool correctly (no tool_call_success in tool log)');
    }

    // 6. Screenshot returned to agent's tool call
    log("Checking for screenshot in tool log...");
    const screenshotOk = toolLogEntries.some(e => /image_b64/.test(e));
    if (!screenshotOk) {
      log("FAIL: No image_b64 found in tool log");
      errors.push('FAIL: Code did not return screenshot to the agent\'s tool call (no image_b64 in tool log)');
    }

    // Print summary
    if (errors.length === 0) {
      log('✅ All checks passed.');
      console.log('✅ All checks passed.');
    } else {
      errors.forEach(e => {
        log(e);
        console.error(e);
      });
      process.exit(1);
    }
  } catch (e) {
    log(`Test script error: ${e.stack}`);
    console.error('Test script error:', e);
    process.exit(1);
  } finally {
    if (browser) {
      log("Closing browser");
      await browser.close();
    }
  }
})(); 