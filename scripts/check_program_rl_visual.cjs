// Read-only browser QA for the actual coding-RL replay; no model calls.
const fs=require('node:fs'),path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require('/Users/mitchellcatoen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
  const directory=path.resolve('runs/program-local-27b-compact-visual');
  const payload=JSON.parse(fs.readFileSync(path.join(directory,'payload.json'),'utf8'));
  const browser=await chromium.launch({headless:true,channel:'chrome'}),errors=[],reports=[];
  try {
    const page=await browser.newPage();page.on('pageerror',error=>errors.push(error.message));
    for(const width of [736,360,320]) {
      await page.setViewportSize({width,height:2200});
      await page.goto(pathToFileURL(path.join(directory,'preview.html')).href);
      const frame=page.frameLocator('iframe');
      await frame.locator('#cr-match option').first().waitFor({state:'attached'});
      const report=await frame.locator('#coding-rl-replay').evaluate((root,payload)=>{
        const data=JSON.parse(root.querySelector('#cr-data').textContent),get=id=>root.querySelector('#cr-'+id);
        const assert=(ok,msg)=>{if(!ok)throw new Error(msg+' / '+root.dataset.episode+' / '+root.dataset.step);};
        assert(JSON.stringify(data)===JSON.stringify(payload),'Embedded payload differs');
        assert(data.optimizer_steps===1&&data.training_tokens===3191,'Update attribution differs');
        assert(data.checkpoints.base.adapter_sha256!==data.checkpoints.step1.adapter_sha256,'Checkpoints collapsed');
        assert(data.same_evaluation_program,'Sampled-program equality omitted');
        assert(get('sample').textContent.includes('identical sampled evaluation code'),'Missing identical-code caveat');
        assert(get('context').textContent.includes('Frozen practice-feedback context'),'Context confused with RL');
        assert(get('authorship').children.length===4,'Training group incomplete');
        assert([...get('authorship').children].filter(n=>n.textContent.includes('no game')).length===2,'Invalid authorship lost');
        assert(data.records_manifest.length===data.episodes.length,'Replay provenance omitted');
        let states=0;
        const phases={};
        for(let i=0;i<data.episodes.length;i++) {
          const e=data.episodes[i];phases[e.phase]=(phases[e.phase]||0)+1;
          get('match').value=i;get('match').dispatchEvent(new Event('change',{bubbles:true}));
          assert(root.dataset.episode===e.id,'Selected episode mismatch');
          assert(data.records_manifest.some(m=>m.episode_id===e.id&&m.record_sha256===e.record_sha256),'Source record mismatch');
          for(const id of e.identities)assert(!data.training_rows.some(r=>r.id===id&&!r.valid),'Invalid draw acquired a game');
          for(let step=0;step<e.states.length;step++) {
            get('round').value=step;get('round').dispatchEvent(new Event('input',{bubbles:true}));
            assert(root.dataset.step===String(step),'Step mismatch');
            assert(get('prev').disabled===(step===0)&&get('next').disabled===(step===e.states.length-1),'Navigation boundary differs');
            for(let p=0;p<2;p++) {
              const r=e.states[step].robots[p],c=data.controllers[e.identities[p]],mark=get('grid').querySelector(`[data-robot="${p}"]`);
              assert(get('values-'+p).textContent===`${r.score} points · Cargo ${r.cargo.length}/3 · Energy ${r.energy}/10`,'Account state differs');
              assert(get('identity-'+p).textContent.includes(c.label),'Program/checkpoint label differs');
              assert(c.checkpoint===e.checkpoint_labels[p]&&c.ir_sha256===e.digests[p],'Identity binding differs');
              assert(Number(mark.dataset.x)===r.x&&Number(mark.dataset.y)===r.y,'Robot position differs');
              if(step) {
                assert(get('action-'+p).textContent.endsWith(e.outcomes[step-1][p].replaceAll('_',' ')),'Action outcome differs');
                assert(get('fault-'+p).textContent.includes(e.decisions[step-1][p].fault?'Fault:':'no interpreter fault'),'Fault state differs');
              }
            }
            for(const site of e.states[step].sites)assert(get('grid').querySelector(`[data-site="${site.x},${site.y}"]`).textContent===site.units+'×'+site.value,'Resource stock differs');
            assert(root.scrollWidth<=root.clientWidth+1,'Horizontal overflow');
            assert(get('grid').querySelectorAll('[data-cell]').length===49,'Grid incomplete');
            const scale=get('grid').getBoundingClientRect().width/get('grid').viewBox.baseVal.width;
            for(const text of get('grid').querySelectorAll('text'))assert(parseFloat(getComputedStyle(text).fontSize)*scale>=11,'Small grid label');
            states++;
          }
          assert(get('result').textContent.startsWith('Final points'),'Terminal score missing');
          get('prev').click();assert(root.dataset.step==='47','Previous failed');
          get('next').click();assert(root.dataset.step==='48','Next failed');
          get('program-details').open=true;
          const expected=e.identities.map(id=>data.controllers[id].python_source).filter(Boolean);
          assert(JSON.stringify([...get('programs').querySelectorAll('code')].map(n=>n.textContent))===JSON.stringify(expected),'Displayed code differs');
          assert(root.scrollWidth<=root.clientWidth+1,'Expanded code overflows');
          get('program-details').open=false;
        }
        assert(JSON.stringify(phases)===JSON.stringify({training:4,before:6,after:6}),'Phase coverage differs');
        return {episodes:data.episodes.length,states,phases,width:root.clientWidth,scrollWidth:root.scrollWidth};
      },payload);
      for(const phase of ['training','before','after']) {
        const index=payload.episodes.findIndex(e=>e.phase===phase);
        await frame.locator('#cr-match').selectOption(String(index));
        await frame.locator('#cr-round').evaluate(n=>{n.value='24';n.dispatchEvent(new Event('input',{bubbles:true}));});
        await page.locator('iframe').screenshot({path:path.join(directory,`${phase}-${width}.png`)});
      }
      reports.push({viewportWidth:width,...report});console.log(JSON.stringify(reports.at(-1)));
    }
    await page.emulateMedia({colorScheme:'dark'});
    await page.setViewportSize({width:736,height:2200});
    await page.goto(pathToFileURL(path.join(directory,'preview.html')).href);
    const frame=page.frameLocator('iframe');
    await frame.locator('#cr-match option').first().waitFor({state:'attached'});
    await frame.locator('#cr-match').selectOption(String(payload.episodes.findIndex(e=>e.phase==='after')));
    await frame.locator('#cr-round').evaluate(n=>{n.value='48';n.dispatchEvent(new Event('input',{bubbles:true}));});
    await page.locator('iframe').screenshot({path:path.join(directory,'after-dark.png')});
    if(errors.length)throw new Error(errors.join('; '));
    fs.writeFileSync(path.join(directory,'qa-report.json'),JSON.stringify({reports,javascriptErrors:errors},null,2)+'\n');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
