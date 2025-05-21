// run_agent_ui_test.mjs
//
// Usage:  node run_agent_ui_test.mjs
//   or    UI_URL=http://127.0.0.1:9999 node run_agent_ui_test.mjs
//
// Needs:  • Node ≥ 18  • "puppeteer" installed  • Python script beside this file
//         • On Windows use "python" instead of "python3" below if needed
// ------------------------------------------------------------

import { spawn }   from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import fetch       from 'node-fetch';          // built-in in Node ≥ 18
import puppeteer   from 'puppeteer';

const PY_SCRIPT = './onefile_fullstack_agent.py';   // adapt path if different
const UI_URL    = process.env.UI_URL ?? 'http://127.0.0.1:7860/';
const HEADLESS  = (process.env.HEADLESS ?? 'true').toLowerCase() !== 'false';

const agent = spawn('python3', [PY_SCRIPT], {
  stdio: ['ignore', 'inherit', 'inherit'], // inherit stdout/stderr so you see them
  env: { ...process.env, PYTHONUNBUFFERED: '1' }     // real-time logs
});

console.log('[BOOT] started python child pid', agent.pid);

// ---------- wait for the HTTP server ----------
const waitForHTTP = async (url, timeoutMs = 25_000) => {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(url, { method: 'GET' });
      if (r.ok) return;
    } catch { /* ignore until timeout */ }
    await delay(600);
  }
  throw new Error(`UI did not respond with 200 OK within ${timeoutMs/1000}s`);
};

try {
  await waitForHTTP(UI_URL);
  console.log('[BOOT] UI is answering HTTP 👍  – launching Chromium …');
} catch (e) {
  console.error('[BOOT] '+e.message);
  agent.kill('SIGTERM');
  process.exit(1);
}

// ---------- Puppeteer ----------
const browser = await puppeteer.launch({
  headless: HEADLESS,
  defaultViewport: null,
  args: ['--disable-web-security']
});
const page = await browser.newPage();

// —— collectors ——
const issues = [];
const pushIssue = (kind, payload) => issues.push({ kind, ...payload });

page.on('console', msg => {
  if (msg.type() === 'error') pushIssue('console', { txt: msg.text() });
});
page.on('pageerror', err => pushIssue('pageerror', { txt: err.message }));
page.on('requestfailed', req =>
  pushIssue('requestfailed', { url: req.url(), err: req.failure()?.errorText }));
page.on('response', res => {
  if (res.status() >= 400)
    pushIssue('http', { url: res.url(), status: res.status() });
});

await page.goto(UI_URL, { waitUntil: 'domcontentloaded' });
await page.waitForSelector('#message-input', { timeout: 10_000 });

// quick noVNC smoke-test (has a child canvas 👀)
await page.waitForSelector('#vnc-canvas', { visible: true, timeout: 8_000 });
const okCanvas = await page.evaluate(() => {
  const v = document.getElementById('vnc-canvas');
  return v && (v.children.length > 0 || v.querySelector('canvas'));
});
if (!okCanvas) {
  console.error('[FAIL] #vnc-canvas visible but empty – noVNC not booted?');
  pushIssue('vnc_canvas_empty', {});
}

// optional: ensure socket.io connected
try {
  await page.waitForFunction(() => window.chatSocket?.connected === true,
                             { timeout: 6_000 });
  console.log('[BOOT] chatSocket connected 👍');
} catch {
  pushIssue('socket_not_connected', {});
}

// ---------- send a prompt ----------
const TEST_PROMPT =
  'Try running each of your tools exactly once; ignore any errors and continue. '
+ 'After you attempted them all, list which worked and which failed. '
+ 'Do NOT simulate tool calls – only real ones.';
await page.type('#message-input', TEST_PROMPT);
await page.keyboard.press('Enter');

console.log('[RUN] prompt sent – waiting 60 s for agent to work …');
await delay(60_000);

// ---------- harvest chat ----------
const chat = await page.evaluate(() => {
  return [...document.querySelectorAll('#chat-output .message')].map(el => ({
    role   : el.querySelector('strong')?.innerText.replace(':','').trim(),
    content: el.querySelector('pre')?.innerText.trim()
  }));
});

// ---------- report ----------
console.log('\n────────── CHAT TRANSCRIPT ──────────');
if (chat.length) chat.forEach((m,i) =>
  console.log(`[${i+1}] ${m.role}: ${m.content ?? '<no pre>'}`));
else console.log('(no messages captured)');

console.log('\n────────── ISSUE SUMMARY ──────────');
if (issues.length === 0)   console.log('✅  no client-side issues');
else issues.forEach((it,i)=> console.log(`[${i+1}]`, it));

// ---------- teardown ----------
await browser.close();
agent.kill('SIGTERM');

process.exit( issues.length === 0 ? 0 : 1 );   // make CI go red when issues exist
