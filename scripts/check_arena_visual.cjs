// CPU/browser-only QA for the rendered recorded-arena replay.
// First run the visualize skill's render.py on selfplay-arena.html.
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('/Users/mitchellcatoen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const preview = path.resolve(process.argv[2] || 'runs/arena-preview.html');
  const out = path.resolve(process.argv[3] || 'runs/arena-visual-qa');
  fs.mkdirSync(out, {recursive:true});
  const browser = await chromium.launch({headless:true, channel:'chrome'});
  const errors = [], reports = [];
  try {
    const page = await browser.newPage();
    page.on('pageerror', error => errors.push(error.message));
    for (const width of [736,360,320]) {
      await page.setViewportSize({width,height:1000});
      await page.goto(pathToFileURL(preview).href);
      const frame = page.frameLocator('iframe');
      await frame.locator('#selfplay-arena[data-episode]').waitFor();
      const result = await frame.locator('#selfplay-arena').evaluate(root => {
        const data = JSON.parse(root.querySelector('#sa-data').textContent);
        const get = name => root.querySelector('#sa-'+name);
        const fail = message => {throw new Error(message+' / '+root.dataset.episode+' / '+root.dataset.step);};
        const equal = (a,b,message) => {if(JSON.stringify(a)!==JSON.stringify(b))fail(message);};
        if(!get('run').textContent.includes(data.run))fail('Missing run identity');
        if(data.policy_arithmetic?.activation_arithmetic==='float32'&&!get('run').textContent.includes('FP32 policy'))fail('Missing arithmetic identity');
        const select = (name,value) => {get(name).value=value;get(name).dispatchEvent(new Event('change',{bubbles:true}));};
        let states = 0;
        const coverage = {};
        for (const e of data.episodes) {
          select('game',e.environment); select('checkpoint',e.checkpoint); select('episode',e.id);
          equal(root.dataset.episode,e.id,'Episode selector mismatch');
          equal(get('checkpoint').value,e.checkpoint,'Checkpoint selector mismatch');
          coverage[e.environment] ||= {};
          coverage[e.environment][e.checkpoint] = (coverage[e.environment][e.checkpoint]||0)+1;
          for (let step=0;step<e.states.length;step++) {
            const state=e.states[step],terminal=step===e.states.length-1;
            equal(Number(root.dataset.step),step,'Incorrect current step');
            equal(get('counter').textContent,`${e.environment==='chess'?'Ply':'Round'} ${step} / ${e.actions.length}`,'Counter mismatch');
            equal(get('prev').disabled,step===0,'Previous control boundary');
            equal(get('next').disabled,terminal,'Next control boundary');
            if (root.scrollWidth>root.clientWidth+1) fail('Horizontal overflow');
            if (/NaN|undefined/.test(get('stage').textContent)) fail('Invalid rendered value');
            if (terminal) {
              if (!state.done || !get('result').textContent.includes('reward ')) fail('Missing terminal result');
              for (let p=0;p<2;p++) if(!get('result').textContent.includes(`P${p} ${e.returns[p]>0?'+':''}${e.returns[p].toFixed(3)}`))fail('Reward differs');
            } else if(get('result').textContent.includes('reward ')) fail('Premature terminal reward');
            if (e.environment==='market') {
              const tables=get('stage').querySelectorAll('table');
              equal(tables.length,2,'Missing market book/account table');
              const accounts=[...tables[1].querySelectorAll('tbody tr')].map(row=>[...row.cells].map(c=>c.textContent));
              equal(accounts,[0,1].map(p=>[`P${p}`,String(state.accounts[p].cash),String(state.accounts[p].inventory),String(state.signals[p])]),'Account values differ');
              if(terminal&&!get('stage').textContent.includes(`Settlement ${state.settlement}`))fail('Settlement mismatch');
            } else if(e.environment==='maze') {
              const maps=get('stage').querySelectorAll('svg');
              equal(maps.length,2,'Missing maze board');
              for(let p=0;p<2;p++) {
                const map=maps[p],circle=map.querySelector('circle'),width=map.viewBox.baseVal.width,cell=(width-16)/state.size;
                equal(map.querySelectorAll('rect').length,state.size*state.size,'Maze cell count');
                for(const [attr,want] of [['cx',8+(state.positions[p][1]+.5)*cell],['cy',8+(state.positions[p][0]+.5)*cell]])
                  if(Math.abs(Number(circle.getAttribute(attr))-want)>1e-6)fail('Maze player differs');
                if(!map.getAttribute('aria-label').includes(`row ${state.positions[p][0]}, column ${state.positions[p][1]}`))fail('Maze accessible state differs');
              }
            } else if(e.environment==='chess') {
              const board=get('stage').querySelector('svg');
              if(!board.getAttribute('aria-label').includes(state.fen))fail('Chess FEN differs');
              const pieces=state.fen.split(' ')[0].replace(/[0-9/]/g,'');
              equal(board.querySelectorAll('circle').length,pieces.length,'Chess piece count differs');
              const labels=[...board.querySelectorAll('text')].map(node=>node.textContent).filter(s=>/^[wb][KQRBNP]$/.test(s));
              equal(labels,[...pieces].map(p=>(p===p.toUpperCase()?'w':'b')+p.toUpperCase()),'Chess pieces differ');
            }
            for(const svg of get('stage').querySelectorAll('svg')) {
              const bounds=svg.getBoundingClientRect(),area=get('stage').getBoundingClientRect();
              if(bounds.left<area.left-1||bounds.right>area.right+1)fail('Board overflows available width');
              for(const label of svg.querySelectorAll('text'))
                if(parseFloat(getComputedStyle(label).fontSize)*bounds.width/svg.viewBox.baseVal.width<11)fail('SVG label too small');
            }
            states++;
            if(!terminal)get('next').click();
          }
          get('prev').click();
          equal(Number(root.dataset.step),e.actions.length-1,'Previous control failed');
        }
        const latestCheckpoint=[...new Set(data.episodes.map(e=>e.checkpoint))].filter(s=>/^cycle\d+$/.test(s)).sort((a,b)=>Number(a.slice(5))-Number(b.slice(5))).at(-1);
        for(const game of ['market','maze','chess']) if(!coverage[game]?.[latestCheckpoint])fail('Missing '+latestCheckpoint+' '+game);
        return {episodes:data.episodes.length,states,coverage,latestCheckpoint,width:root.clientWidth,scrollWidth:root.scrollWidth};
      });
      for(const game of ['market','maze','chess']) {
        await frame.locator('#sa-game').selectOption(game);
        await frame.locator('#sa-checkpoint').selectOption(result.latestCheckpoint);
        const name=await frame.locator('#sa-episode option').first().getAttribute('value');
        await frame.locator('#sa-episode').selectOption(name);
        await frame.locator('#selfplay-arena').evaluate(root=>{
          const e=JSON.parse(root.querySelector('#sa-data').textContent).episodes.find(e=>e.id===root.dataset.episode);
          for(let i=0;i<Math.min(4,e.actions.length);i++)root.querySelector('#sa-next').click();
        });
        await page.waitForTimeout(100);
        await page.screenshot({path:path.join(out,`${game}-${result.latestCheckpoint}-${width}.png`),fullPage:true});
      }
      reports.push({viewportWidth:width,...result});
      console.log(JSON.stringify(reports.at(-1)));
    }
    await page.emulateMedia({colorScheme:'dark'});
    await page.setViewportSize({width:736,height:1000});
    await page.goto(pathToFileURL(preview).href);
    const frame=page.frameLocator('iframe');
    await frame.locator('#selfplay-arena[data-episode]').waitFor();
    for(const game of ['market','maze','chess']) {
      await frame.locator('#sa-game').selectOption(game);
      await frame.locator('#sa-checkpoint').selectOption(reports[0].latestCheckpoint);
      await frame.locator('#sa-next').click();
      await page.waitForTimeout(100);
      await page.screenshot({path:path.join(out,`${game}-${reports[0].latestCheckpoint}-dark.png`),fullPage:true});
    }
    if(errors.length)throw new Error(errors.join('; '));
    fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({preview,reports,javascriptErrors:errors},null,2)+'\n');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
