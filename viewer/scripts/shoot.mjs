// Screenshot the running dev server: node scripts/shoot.mjs <out.png> [url]
import { chromium } from 'playwright';

const out = process.argv[2] ?? 'shots/earth.png';
const url = process.argv[3] ?? 'http://localhost:5180/';
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(url);
await page.waitForTimeout(1500);
await page.screenshot({ path: out });
await browser.close();
if (errors.length) { console.error(errors.join('\n')); process.exit(1); }
console.log(`wrote ${out}`);
