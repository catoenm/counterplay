// CPU/browser QA for the rendered Elo fragment. It verifies embedded values,
// all snapshots, plotted coordinates, series toggles, and cross-series hover.
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require('/Users/mitchellcatoen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async()=>{
  const preview=path.resolve(process.argv[2]||'runs/arena-v2/elo-preview.html');
  const out=path.resolve(process.argv[3]||'runs/arena-v2/elo-visual-qa');
  const result=JSON.parse(fs.readFileSync(path.resolve(process.argv[4]||'runs/arena-v2/elo.json'),'utf8'));
  fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({headless:true,channel:'chrome'});
  const errors=[],reports=[];
  try {
    const page=await browser.newPage();
    page.on('pageerror',error=>errors.push(error.message));
    for(const width of [736,360,320]) {
      await page.setViewportSize({width,height:2200});
      await page.goto(pathToFileURL(preview).href);
      const frame=page.frameLocator('iframe');
      await frame.locator('.er-chart').first().waitFor();
      const report=await frame.locator('#arena-rating-history').evaluate((root,result)=>{
        const data=JSON.parse(root.querySelector('#er-data').textContent);
        const get=id=>root.querySelector('#er-'+id);
        const assert=(yes,msg)=>{if(!yes)throw new Error(msg);};
        for(const key of Object.keys(data))assert(JSON.stringify(data[key])===JSON.stringify(result[key]),'Embedded data differs: '+key);
        assert(get('run').textContent.includes(data.run),'Run identity missing');
        assert(data.rating_pool.policy_arithmetic.activation_arithmetic==='float32'&&get('run').textContent.includes('FP32 policy'),'FP32 pool identity missing');
        const games=Object.entries(data.games);
        const display=id=>id==='base'?'Base':id==='heuristic'?'Heuristic':'Checkpoint '+id.slice(5);
        const rank=id=>id==='base'?0:id==='heuristic'?10000:Number(id.slice(5));
        const identities=[...new Set(games.flatMap(([,game])=>Object.keys(game.series)))].sort((a,b)=>rank(a)-rank(b));
        let checkedPoints=0;
        for(const stage of data.observed_iterations) {
          get('stage').value=stage;get('stage').dispatchEvent(new Event('change',{bubbles:true}));
          const sections=[...root.querySelectorAll('.er-game')];
          assert(sections.length===games.length,'Missing environment');
          sections.forEach((section,index)=>{
            const [name,game]=games[index];
            const expected=identities.flatMap(id=>{const p=game.series[id]?.filter(p=>p.iteration<=stage).at(-1);return p?[`${display(id)} ${Math.round(p.rating)} (${p.games} games)`]:[];});
            assert(JSON.stringify([...section.querySelectorAll('.er-values span')].map(n=>n.textContent))===JSON.stringify(expected),'Snapshot values differ '+name+'/'+stage);
            const svg=section.querySelector('svg'),w=svg.viewBox.baseVal.width,h=svg.viewBox.baseVal.height;
            assert(Math.abs(w-section.clientWidth)<1,'Fixed-width SVG');
            assert(svg.querySelectorAll('.axis-title[data-axis]').length===2,'Missing axis titles');
            const svgBounds=svg.getBoundingClientRect();
            for(const node of svg.querySelectorAll('text')) {
              const b=node.getBoundingClientRect();assert(b.left>=svgBounds.left-1&&b.top>=svgBounds.top-1&&b.right<=svgBounds.right+1&&b.bottom<=svgBounds.bottom+1,'Clipped SVG label: '+node.textContent);
              assert(parseFloat(getComputedStyle(node).fontSize)>=11,'Undersized SVG text');
            }
            const ratings=[data.initial_rating,...games.flatMap(([,g])=>Object.values(g.series).flat().map(p=>p.rating))];
            const low=Math.min(...ratings),high=Math.max(...ratings),pad=Math.max(12,(high-low)*.12);
            const xmin=Math.min(...data.observed_iterations),xmax=Math.max(...data.observed_iterations);
            const x=v=>70+(v-xmin)/(xmax-xmin)*(w-90);
            const y=v=>h-54-(v-(low-pad))/(high-low+2*pad)*(h-72);
            for(const [id,points] of Object.entries(game.series)) {
              const nodes=[...svg.querySelectorAll(`circle[data-series="${id}"]`)];
              assert(nodes.length===points.length,'Missing observations '+id);
              nodes.forEach((node,i)=>{assert(Math.abs(Number(node.getAttribute('cx'))-x(points[i].iteration))<1e-5&&Math.abs(Number(node.getAttribute('cy'))-y(points[i].rating))<1e-5,'Plot coordinates differ');checkedPoints++;});
            }
          });
          assert(root.scrollWidth<=root.clientWidth+1,'Horizontal overflow');
        }
        for(const id of identities) {
          const label='Show '+display(id);
          get('legend').querySelector(`[aria-label="${label}"]`).click();
          assert(!root.querySelector(`[data-series="${id}"]`),'Hidden series still plotted');
          assert(get('legend').querySelector(`[aria-label="${label}"]`).getAttribute('aria-pressed')==='false','Toggle state differs');
          get('legend').querySelector(`[aria-label="${label}"]`).click();
        }
        const svg=root.querySelector('.er-chart'),overlay=svg.querySelector('[data-chart-hover-overlay="cross-series"]');
        const box=svg.getBoundingClientRect(),px=70+(svg.viewBox.baseVal.width-90)*.45;
        const event=()=>new PointerEvent('pointermove',{clientX:box.left+px,clientY:box.top+80,bubbles:true});
        overlay.dispatchEvent(event());
        assert(!get('tooltip').hidden,'Hover missing');
        assert(Math.abs(Number(svg.querySelector('[data-chart-hover-guide]').getAttribute('x1'))-px)<.01,'Hover guide snapped away from cursor');
        assert(svg.querySelectorAll('[data-chart-hover-marker]').length>=2,'Cross-series hover missing');
        for(const marker of svg.querySelectorAll('[data-chart-hover-marker]'))assert(Math.abs(Number(marker.getAttribute('cx'))-px)<.01,'Hover markers not aligned');
        overlay.dispatchEvent(new PointerEvent('pointerleave',{bubbles:true}));
        assert(get('tooltip').hidden,'Hover did not clear');
        return {run:data.run,iterations:data.observed_iterations,checkedPoints,identities,viewportWidth:window.innerWidth,width:root.clientWidth,scrollWidth:root.scrollWidth};
      },result);
      reports.push(report);
      await page.locator('iframe').screenshot({path:path.join(out,`ratings-${width}.png`)});
      console.log(JSON.stringify(report));
    }
    await page.emulateMedia({colorScheme:'dark'});
    await page.setViewportSize({width:736,height:2200});
    await page.goto(pathToFileURL(preview).href);
    await page.frameLocator('iframe').locator('.er-chart').first().waitFor();
    await page.locator('iframe').screenshot({path:path.join(out,'ratings-dark.png')});
    if(errors.length)throw new Error(errors.join('; '));
    fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({preview,reports,javascriptErrors:errors},null,2)+'\n');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
