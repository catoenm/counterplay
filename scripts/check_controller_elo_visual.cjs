const fs=require('node:fs'),path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require('/Users/mitchellcatoen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
  const directory=path.resolve('runs/controller-tournament-visual'),expected=JSON.parse(fs.readFileSync(path.join(directory,'payload.json'),'utf8'));
  const browser=await chromium.launch({channel:'chrome',headless:true}),errors=[],reports=[];
  try{
    const page=await browser.newPage();page.on('pageerror',error=>errors.push(error.message));
    for(const width of [736,360,320]){
      await page.setViewportSize({width,height:1800});
      await page.goto(pathToFileURL(path.join(directory,'preview.html')).href);
      const frame=page.frameLocator('iframe');await frame.locator('#ct-chart path').first().waitFor();
      const report=await frame.locator('#controller-tournament-rating-history').evaluate((root,expected)=>{
        const get=id=>root.querySelector('#ct-'+id),data=JSON.parse(get('data').textContent),assert=(ok,msg)=>{if(!ok)throw new Error(msg);};
        assert(JSON.stringify(data)===JSON.stringify(expected),'Embedded data differs from verified payload');
        assert(data.games===252&&data.identities.length===7&&data.records.length===252,'Wrong tournament coverage');
        assert(root.textContent.includes('Frozen authoring versions')&&root.textContent.includes('not average model ability'),'Identity caveat absent');
        assert(root.scrollWidth<=root.clientWidth+1,'Horizontal overflow');
        const svg=get('chart'),w=svg.viewBox.baseVal.width,h=svg.viewBox.baseVal.height;
        assert(w===root.clientWidth,'Fixed-width chart');
        const extent=d3.extent(Object.values(data.series).flat().map(p=>p.rating)),pad=Math.max(12,(extent[1]-extent[0])*.1);
        const x=d3.scaleLinear().domain([0,252]).range([70,w-20]),y=d3.scaleLinear().domain([extent[0]-pad,extent[1]+pad]).range([h-56,18]);
        for(const id of data.identities){
          const points=data.series[id],line=svg.querySelector(`path[data-series="${id}"]`),dot=svg.querySelector(`[data-final="${id}"]`);
          assert(points.length===253&&points[0].rating===1500&&points.at(-1).played===72,'Missing rating observations');
          assert(line.getAttribute('d')===d3.line().x(p=>x(p.game)).y(p=>y(p.rating))(points),'Plotted series differs');
          assert(Math.abs(Number(dot.getAttribute('cx'))-x(252))<1e-7&&Math.abs(Number(dot.getAttribute('cy'))-y(points.at(-1).rating))<1e-7,'Final marker differs');
          assert(get('legend').querySelector(`[data-identity="${id}"]`).textContent.includes(String(Math.round(data.controllers[id].rating))),'Final visible value differs');
        }
        const area=svg.getBoundingClientRect(),labels=[...svg.querySelectorAll('text')];
        for(const label of labels){const b=label.getBoundingClientRect();assert(b.left>=area.left-1&&b.right<=area.right+1&&b.top>=area.top-1&&b.bottom<=area.bottom+1,'Clipped label: '+label.textContent);assert(parseFloat(getComputedStyle(label).fontSize)>=11,'Small text');}
        for(let i=0;i<labels.length;i++)for(let j=i+1;j<labels.length;j++){const a=labels[i].getBoundingClientRect(),b=labels[j].getBoundingClientRect();assert(a.right+4<=b.left||b.right+4<=a.left||a.bottom+4<=b.top||b.bottom+4<=a.top,'Overlapping axis labels');}
        if(w<=360)assert(svg.querySelectorAll('[data-x-tick]').length<=4,'Too many narrow ticks');
        for(const id of data.identities){get('legend').querySelector(`[data-identity="${id}"]`).click();assert(!get('chart').querySelector(`[data-series="${id}"]`),'Hidden controller still plotted');get('legend').querySelector(`[data-identity="${id}"]`).click();}
        const current=get('chart'),overlay=current.querySelector('[data-chart-hover-overlay="cross-series"]'),box=current.getBoundingClientRect(),game=87.25,px=x(game);
        const pointer=type=>new PointerEvent(type,{clientX:box.left+px,clientY:box.top+90,bubbles:true});overlay.dispatchEvent(pointer('pointermove'));
        assert(!get('tooltip').hidden,'Hover tooltip absent');
        assert(current.querySelectorAll('[data-chart-hover-marker]').length===7,'Missing cross-series hover values');
        assert(Math.abs(Number(current.querySelector('[data-chart-hover-guide]').getAttribute('x1'))-px)<.01,'Guide snapped away from cursor');
        for(const id of data.identities){const points=data.series[id],rating=points[87].rating+.25*(points[88].rating-points[87].rating),mark=current.querySelector(`[data-chart-hover-marker="${id}"]`);assert(get('tooltip').textContent.includes(data.controllers[id].label+': '+rating.toFixed(1)),'Interpolated tooltip value differs');assert(Math.abs(Number(mark.getAttribute('cy'))-y(rating))<.01,'Hover marker differs');}
        overlay.dispatchEvent(pointer('click'));overlay.dispatchEvent(pointer('pointerleave'));assert(!get('tooltip').hidden,'Touch pin did not persist');overlay.dispatchEvent(pointer('click'));assert(get('tooltip').hidden,'Touch pin did not clear');
        return {controllers:7,matches:252,points:7*253,width:root.clientWidth,scrollWidth:root.scrollWidth};
      },expected);
      await page.locator('iframe').screenshot({path:path.join(directory,`ratings-${width}.png`)});
      reports.push({viewportWidth:width,...report});console.log(JSON.stringify(reports.at(-1)));
    }
    await page.emulateMedia({colorScheme:'dark'});await page.setViewportSize({width:736,height:1800});
    await page.goto(pathToFileURL(path.join(directory,'preview.html')).href);await page.frameLocator('iframe').locator('#ct-chart path').first().waitFor();
    await page.locator('iframe').screenshot({path:path.join(directory,'ratings-dark.png')});
    if(errors.length)throw new Error(errors.join('; '));
    fs.writeFileSync(path.join(directory,'qa-report.json'),JSON.stringify({reports,javascriptErrors:errors},null,2)+'\n');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
