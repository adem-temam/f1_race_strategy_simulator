const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/usr/bin/google-chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu']
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });

  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.message));

  console.log('Navigating to http://127.0.0.1:8888/ ...');
  await page.goto('http://127.0.0.1:8888/', { waitUntil: 'networkidle' });

  console.log('Clicking Simulate Strategies button...');
  await page.click('#runSimulationBtn');

  // Wait for results table to populate with P1 tag
  console.log('Waiting for simulation results...');
  await page.waitForSelector('.tag-p1', { timeout: 10000 });

  // Wait 1s for Chart.js animation
  await page.waitForTimeout(1000);

  const outPath = path.resolve(__dirname, '../docs/assets/dashboard_preview.png');
  await page.screenshot({ path: outPath });
  console.log('Successfully captured screenshot to', outPath);

  await browser.close();
})();
