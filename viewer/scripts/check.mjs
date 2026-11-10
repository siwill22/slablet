// Exercise the Earth viewer: projection button, every field and arrow set, depths, ages; fails on any page error.
// node scripts/check.mjs [url]
import { chromium } from 'playwright';

const url = process.argv[2] ?? 'http://localhost:5180/';
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(url);
await page.waitForTimeout(1500);
for (let k = 0; k < 3; k++) { await page.click('#projection-toggle'); await page.waitForTimeout(300); }
const selects = page.locator('.lil-gui select');
const fields = await selects.nth(0).locator('option').allTextContents();
const vectors = await selects.nth(1).locator('option').allTextContents();
const numberInput = (label) => page.locator('.lil-gui .lil-controller, .lil-gui .controller')
  .filter({ hasText: label }).locator('input').first();
const scrub = page.locator('.asth-scrub');
const n = Number(await scrub.getAttribute('max'));
let visits = 0;
for (const f of fields) {
  await selects.nth(0).selectOption({ label: f });
  for (const i of [0, Math.floor(n / 2), n]) { await scrub.fill(String(i)); await page.waitForTimeout(80); visits++; }
}
for (const v of vectors) {
  await selects.nth(1).selectOption({ label: v });
  for (const i of [0, n]) { await scrub.fill(String(i)); await page.waitForTimeout(80); visits++; }
}
// Each depth control is live only while the field or arrows use its set.
const depthChecks = [['Below the plate (km)', 'below the plate', ['25', '150', '400']],
  ['Depth (km)', 'Shear-velocity anomaly (model)', ['50', '1450', '2850']]];
for (const [label, fieldText, depths] of depthChecks) {
  await selects.nth(0).selectOption({ label: fields.find((f) => f.includes(fieldText)) });
  for (const d of depths) {
    await numberInput(label).fill(d); await page.keyboard.press('Enter'); await page.waitForTimeout(80); visits++;
  }
}
for (let i = 0; i <= n; i += 1) { await scrub.fill(String(i)); await page.waitForTimeout(30); visits++; }
await browser.close();
console.log(`${fields.length} fields, ${vectors.length} arrow sets, ${visits} views, ${errors.length} errors`);
if (errors.length) { console.error(errors.join('\n')); process.exit(1); }
