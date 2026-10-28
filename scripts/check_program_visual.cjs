const {chromium}=require('/Users/mitchellcatoen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,channel:'chrome'}),errors=[];
  try{
    const page=await browser.newPage();page.on('pageerror',e=>errors.push(e.message));
    for(const width of [736,360,320]){
      await page.setViewportSize({width,height:900});
      await page.goto('file:///Users/mitchellcatoen/Documents/workspace/selfplay/runs/program-arena-preview.html');
      const frame=page.frameLocator('iframe');
      await frame.locator('#pa-match option').first().waitFor({state:'attached'});
      const n=await frame.locator('#pa-match option').count();
      for(let i=0;i<n;i++){
        await frame.locator('#pa-match').selectOption(String(i));
        await frame.locator('#pa-next').click();
        if(!/^Round 1 \//.test(await frame.locator('#pa-step').innerText()))throw new Error('Next control failed');
        await frame.locator('#pa-prev').click();
        if(!/^Round 0 \//.test(await frame.locator('#pa-step').innerText()))throw new Error('Previous control failed');
        const checked=await frame.locator('#program-arena-replay').evaluate((root,index)=>{
          const data=JSON.parse(root.querySelector('#pa-data').textContent),e=data.episodes[index],slider=root.querySelector('#pa-round');
          if(data.records_manifest?.length!==data.total)throw new Error('Missing all-record provenance');
          if(!data.records_manifest.some(m=>m.episode_id===e.id&&m.record_sha256===e.record_sha256))throw new Error('Displayed record is absent from provenance');
          if(!root.querySelector('#pa-source').textContent.includes(e.source))throw new Error('Missing source identity');
          for(let step=0;step<e.states.length;step++){
            slider.value=String(step);slider.dispatchEvent(new Event('input',{bubbles:true}));
            for(let p=0;p<2;p++){
              const r=e.states[step].robots[p];
              if(root.querySelector('#pa-values-'+p).textContent!==`${r.score} points · Cargo ${r.cargo.length}/3 · Energy ${r.energy}/10`)throw new Error('Account state differs');
              if(!root.querySelector('#pa-identity-'+p).textContent.includes(e.players[p]))throw new Error('Program identity differs');
              if(step&&!root.querySelector('#pa-action-'+p).textContent.endsWith(e.outcomes[step-1][p].replaceAll('_',' ')))throw new Error('Outcome differs');
              if(step&&e.decisions[step-1][p].fault&&!root.querySelector('#pa-fault-'+p).textContent.includes('Fault:'))throw new Error('Interpreter fault is hidden');
            }
            if(root.scrollWidth>root.clientWidth+1)throw new Error('Horizontal overflow at round '+step);
          }
          return e.states.length;
        },i);
        if(!(await frame.locator('#pa-result').innerText()).startsWith('Final points'))throw new Error('Missing terminal result');
        if(await frame.locator('#pa-next').isEnabled())throw new Error('Next remains enabled at terminal state');
        if(!checked)throw new Error('No saved states');
      }
      await frame.locator('#pa-match').selectOption('0');
      await frame.locator('#pa-round').evaluate(el=>{el.value='16';el.dispatchEvent(new Event('input',{bubbles:true}));});
      await page.waitForTimeout(250);
      await page.screenshot({path:`/Users/mitchellcatoen/Documents/workspace/selfplay/runs/program-arena-visual-${width}.png`,fullPage:true});
      const peer=await frame.locator('#program-arena-replay').evaluate(root=>JSON.parse(root.querySelector('#pa-data').textContent).episodes.findIndex(e=>e.identities.includes('qwen38-27b-1')&&e.identities.includes('qwen38-27b-2')));
      if(peer<0)throw new Error('Original failed peer programs were omitted');
      await frame.locator('#pa-match').selectOption(String(peer));
      await frame.locator('#pa-round').evaluate(el=>{el.value=el.max;el.dispatchEvent(new Event('input',{bubbles:true}));});
      if(!(await frame.locator('#pa-result').innerText()).startsWith('Final points 0–0'))throw new Error('Original Qwen peer deadlock result differs');
      await page.screenshot({path:`/Users/mitchellcatoen/Documents/workspace/selfplay/runs/program-arena-peer-${width}.png`,fullPage:true});
      const revised=await frame.locator('#program-arena-replay').evaluate(root=>JSON.parse(root.querySelector('#pa-data').textContent).episodes.findIndex(e=>e.identities.includes('controller-python-1')&&e.identities.includes('qwen38-27b-1-r2')));
      if(revised<0)throw new Error('Revised Qwen vs DeepSeek peer game missing');
      await frame.locator('#pa-match').selectOption(String(revised));
      await frame.locator('#pa-round').evaluate(el=>{el.value='16';el.dispatchEvent(new Event('input',{bubbles:true}));});
      if(!(await frame.locator('#pa-robots').innerText()).includes('feedback 2'))throw new Error('Authoring revision is not labeled');
      await page.screenshot({path:`/Users/mitchellcatoen/Documents/workspace/selfplay/runs/program-arena-revised-${width}.png`,fullPage:true});
      const fault=await frame.locator('#program-arena-replay').evaluate(root=>JSON.parse(root.querySelector('#pa-data').textContent).episodes.findIndex(e=>e.identities.includes('qwen38-27b-2-r2')));
      if(fault<0)throw new Error('Broken final revision omitted');
      await frame.locator('#pa-match').selectOption(String(fault));
      await frame.locator('#pa-round').evaluate(el=>{el.value='8';el.dispatchEvent(new Event('input',{bubbles:true}));});
      if(!(await frame.locator('#pa-robots').innerText()).includes('Fault: invalid move'))throw new Error('Broken revision fault is not visible');
      await page.screenshot({path:`/Users/mitchellcatoen/Documents/workspace/selfplay/runs/program-arena-fault-${width}.png`,fullPage:true});
      const bounds=await frame.locator('#program-arena-replay').evaluate(root=>({width:root.clientWidth,scroll:root.scrollWidth,height:root.scrollHeight}));
      console.log(JSON.stringify({width,episodes:n,bounds}));
    }
    await page.emulateMedia({colorScheme:'dark'});
    await page.setViewportSize({width:736,height:950});
    await page.goto('file:///Users/mitchellcatoen/Documents/workspace/selfplay/runs/program-arena-preview.html');
    const darkFrame=page.frameLocator('iframe');
    await darkFrame.locator('#pa-match option').first().waitFor({state:'attached'});
    await darkFrame.locator('#pa-round').evaluate(el=>{el.value='16';el.dispatchEvent(new Event('input',{bubbles:true}));});
    await page.waitForTimeout(250);
    await page.screenshot({path:'/Users/mitchellcatoen/Documents/workspace/selfplay/runs/program-arena-visual-dark.png',fullPage:true});
    if(errors.length)throw new Error(errors.join('; '));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
