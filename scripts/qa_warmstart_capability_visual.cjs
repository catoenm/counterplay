// Reporting-only browser QA; reads embedded recorded states, never runs models.
const {chromium}=require('/Users/mitchellcatoen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const root=path.resolve(__dirname,'..');
const run=path.join(root,'runs/cardinal-warmstart-gpu-v2');
const fragment='/Users/mitchellcatoen/.codex/visualizations/2026/10/20/01a15b4f-05a3-7053-9034-4d1bc9f8d726/cardinal-warmstart-capability-replay.html';
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
  const output=path.join(run,'capability-replay-qa.json');
  if(fs.existsSync(output))throw Error('Refusing to overwrite QA');
  const browser=await chromium.launch({headless:true,channel:'chrome'});
  const results=[];
  for(const [width,theme] of [[736,'light'],[360,'light'],[320,'light'],[736,'dark']]){
    const context=await browser.newContext({viewport:{width:width+32,height:1200},colorScheme:theme});
    const page=await context.newPage();const errors=[];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.goto('file://'+path.join(run,'capability-replay-preview.html'));
    await page.waitForTimeout(250);
    const frame=page.frames().find(f=>f.parentFrame());
    const result=await frame.evaluate(()=>{
      const root=document.getElementById('cardinal-warmstart-capability-replay');
      const data=JSON.parse(root.querySelector('#pa-data').textContent);
      const get=id=>root.querySelector('#'+id),select=get('pa-match'),slider=get('pa-round');
      const check=(ok,msg)=>{if(!ok)throw Error(msg);};
      check(data.episodes.length===10,'Match count');
      check(JSON.stringify(data.episodes.map(e=>e.sample_index))==='[0,0,1,1,5,5,6,6,7,7]','Chronological order');
      check(select.value==='0' && slider.value==='0','Default is first chronological sample');
      check(get('pa-source').textContent==='32 supervised updates · 0 RL updates · Calibration only, no heldout comparison','Training label');
      check(get('pa-coverage').textContent===data.coverage_label,'Coverage label');
      let checked=0;
      const actionText=a=>a.action==='move'?'Move '+(a.dx===1?'east':a.dx===-1?'west':a.dy===1?'south':'north'):a.action[0].toUpperCase()+a.action.slice(1);
      data.episodes.forEach((e,i)=>{
        select.value=String(i);select.dispatchEvent(new Event('change',{bubbles:true}));
        check(slider.value==='0','Selecting a match resets round');
        e.states.forEach((state,s)=>{
          slider.value=String(s);slider.dispatchEvent(new Event('input',{bubbles:true}));
          check(get('pa-step').textContent===`Round ${s} / ${e.states.length-1}`,'Round label');
          check(get('pa-prev').disabled===(s===0) && get('pa-next').disabled===(s===e.states.length-1),'Boundary buttons');
          state.robots.forEach((r,p)=>{
            check(get('pa-values-'+p).textContent===`${r.score} points · Cargo ${r.cargo.length}/3 · Energy ${r.energy}/10`,'Robot metrics');
            check(get('pa-identity-'+p).textContent.includes(e.players[p]),'Model and bot identity');
            const d=s?e.decisions[s-1][p]:null;
            check(get('pa-action-'+p).textContent===(d?`${actionText(d.action)} → ${e.outcomes[s-1][p].replaceAll('_',' ')}`:'Before the first action'),'Action/outcome');
            check(get('pa-fault-'+p).textContent===(d?`${d.steps} operations · ${d.fault?'Fault: '+d.fault.replaceAll('_',' '):'no interpreter fault'}`:'Memory starts at zero'),'Fault detail');
          });
          check(root.scrollWidth<=root.clientWidth+1,'Root overflow');
          check(document.documentElement.scrollWidth<=document.documentElement.clientWidth+1,'Page overflow');
          checked++;
        });
      });
      select.value='0';select.dispatchEvent(new Event('change',{bubbles:true}));
      get('pa-next').click();check(slider.value==='1','Next button');
      get('pa-prev').click();check(slider.value==='0','Previous button');
      // Inspect an actual later state, rather than only an untouched board.
      slider.value='24';slider.dispatchEvent(new Event('input',{bubbles:true}));
      return {games:data.episodes.length,states_checked:checked,content_width:root.clientWidth,default_sample:0,shown_round:24};
    });
    await page.waitForTimeout(100);
    const screenshot=path.join(run,`capability-replay-qa-${width}-${theme}.png`);
    await frame.locator('#cardinal-warmstart-capability-replay').screenshot({path:screenshot});
    if(errors.length)throw Error(errors.join('\n'));
    results.push({width,theme,...result,page_errors:errors,screenshot,sha256:sha(screenshot)});
    await context.close();
  }
  await browser.close();
  fs.writeFileSync(output,JSON.stringify({kind:'warmstart_capability_replay_browser_qa',fragment,fragment_sha256:sha(fragment),
    audit_sha256:sha(path.join(run,'final-audit.json')),provenance_sha256:sha(path.join(run,'capability-replay-provenance.json')),
    passed:true,results},null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({passed:true,output,results}));
})().catch(e=>{console.error(e);process.exit(1);});
