const { chromium } = require('/Users/mitchellcatoen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const browser = await chromium.launch({headless:true,channel:'chrome'});
  const errors=[];
  try {
    const page=await browser.newPage();
    page.on('pageerror',e=>errors.push(e.message));
    for (const width of [736,360,320]) {
      await page.setViewportSize({width,height:850});
      await page.goto('file:///Users/mitchellcatoen/Documents/workspace/selfplay/runs/challenge-preview.html');
      const frame=page.frameLocator('iframe');
      await frame.locator('#ac-trace option').first().waitFor({state:'attached'});
      const count=await frame.locator('#ac-trace option').count();
      for(let i=0;i<count;i++){
        await frame.locator('#ac-trace').selectOption(String(i));
        while(await frame.locator('#ac-next').isEnabled())await frame.locator('#ac-next').click();
        const label=await frame.locator('#ac-result').innerText();
        if(!label.startsWith('Arbiter: '))throw new Error('Missing arbiter result');
      }
      const overflow=await frame.locator('#agent-challenge').evaluate(el=>el.scrollWidth>el.clientWidth+1);
      if(overflow)throw new Error('Overflow at '+width);
      await page.waitForTimeout(250);
      await page.screenshot({path:`/Users/mitchellcatoen/Documents/workspace/selfplay/runs/challenge-visual-${width}.png`,fullPage:true});
      console.log(JSON.stringify({width,traces:count,overflow}));
    }
    if(errors.length)throw new Error(errors.join('; '));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
