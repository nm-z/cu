// test_agent_ui.js
// Usage:  UI_URL=http://127.0.0.1:7860 node test_agent_ui.js
// -----------------------------------------------------------

import puppeteer from 'puppeteer';

const UI_URL  = process.env.UI_URL  || 'http://127.0.0.1:7860/';
const HEADLESS = (process.env.HEADLESS ?? 'true').toLowerCase() !== 'false'; // HEADLESS=false to see the window

(async () => {
    const browser = await puppeteer.launch({
        headless: HEADLESS,
        defaultViewport: null,
            args: ['--disable-web-security'] // helps if you do weird localhost CORS things
    });
    const page = await browser.newPage();
    
    // ───────────– collectors ───────────
    const issues = [];
    
    page.on('console', msg => {
        if (msg.type() === 'error') {
            const txt = msg.text();
            console.error('[console.error]', txt);
            issues.push({kind: 'console', txt});
        }
    });
    
    page.on('pageerror', err => {
        console.error('[pageerror]', err.message);
        issues.push({kind: 'pageerror', txt: err.message});
    });
    
    page.on('requestfailed', req => {
        const err = req.failure()?.errorText || 'unknown';
        console.error('[requestfailed]', req.url(), '→', err);
        issues.push({kind: 'requestfailed', url: req.url(), err});
    });
    
    page.on('response', res => {
        if (res.status() >= 400) {
            console.error('[http ' + res.status() + ']', res.url());
            issues.push({kind: 'http', url: res.url(), status: res.status()});
        }
    });
    
    // ───────────– test flow ───────────
    console.log('Opening', UI_URL);
    await page.goto(UI_URL, {waitUntil: 'domcontentloaded'});
    
    // wait for the chat input that exists in your template
    await page.waitForSelector('#message-input', {timeout: 10000});
    
    // Check for VNC canvas visibility and content
    try {
        await page.waitForSelector('#vnc-canvas', { visible: true, timeout: 5000 });
        const vncCanvasHasChildren = await page.evaluate(() => {
            const vncElem = document.getElementById('vnc-canvas');
            // noVNC typically creates a <canvas> element or a div structure inside.
            // Check for any child elements or a canvas specifically.
            return vncElem && (vncElem.children.length > 0 || vncElem.querySelector('canvas'));
        });
        if (vncCanvasHasChildren) {
            console.log('VNC canvas is visible and appears to have content (e.g., noVNC initialized) 👍');
        } else {
            throw new Error('VNC canvas (#vnc-canvas) is visible but has no child elements, noVNC might not be initialized.');
        }
    } catch (e) {
        console.error('⚠️ VNC canvas check failed:', e.message);
        issues.push({kind: 'vnc_check_failed', txt: e.message});
        await browser.close();
        process.exit(1); // Exit immediately if VNC check fails
    }
    
    // OPTIONAL: verify that socket.io actually connected
    try {
        await page.waitForFunction(() => {
            // Check the application's specific socket instance
            return window.chatSocket && window.chatSocket.connected;
        }, {timeout: 5000});
        console.log('Socket.IO client (window.chatSocket) appears connected 👍');
    } catch { console.warn('⚠️  Socket.IO client (window.chatSocket) did NOT connect within 5 s'); }
    
    // send a dummy prompt
    const testPrompt = "Try each of your tools, one at a time - the goal is to use each and ignore the error and move onto the next one. The goal is that you only run each of your tools once and don\'t try the same one twice. At the end, tell me which ones worked and which ones did not. DO NOT simulate any tool calls.";
    await page.type('#message-input', testPrompt);
    await page.keyboard.press('Enter');
    
    // wait a moment to let the UI respond / errors surface
    // Increased wait time significantly to allow the agent to perform multiple tool calls
    console.log('Prompt sent. Waiting for 60 seconds for the agent to process and respond...');
    await new Promise(resolve => setTimeout(resolve, 60000)); 
    
    // ───────────– results ───────────
    console.log('\n──────────── summary ────────────');

    // Extract chat messages
    const chatMessages = await page.evaluate(() => {
        const messages = [];
        document.querySelectorAll('#chat-output .message').forEach(msgElement => {
            const roleElement = msgElement.querySelector('strong');
            const contentElement = msgElement.querySelector('pre');
            if (roleElement && contentElement) {
                messages.push({
                    role: roleElement.innerText.replace(':', '').trim(),
                    content: contentElement.innerText.trim()
                });
            }
        });
        return messages;
    });

    console.log('\n──────────── Chat History ────────────');
    if (chatMessages.length > 0) {
        chatMessages.forEach((msg, idx) => {
            console.log(`[Chat ${idx+1}] ${msg.role}: ${msg.content}`);
        });
    } else {
        console.log('No chat messages found in #chat-output.');
    }

    if (issues.length === 0) {
        console.log('✅  No client-side errors captured');
    } else {
        console.log(`❌  Captured ${issues.length} issue(s):`);
        issues.forEach((i, idx) => console.log(`[${idx+1}]`, i));
    }
    
    await browser.close();
})();
