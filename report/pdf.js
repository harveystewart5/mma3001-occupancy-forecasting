const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage();
await p.goto('file://'+process.argv[2],{waitUntil:'networkidle'});
await p.pdf({path:process.argv[3],format:'A4',printBackground:true,displayHeaderFooter:true,
 headerTemplate:'<div></div>',
 footerTemplate:'<div style="font-size:8pt;width:100%;text-align:center;color:#666;">MMA3001 Project Report — Harvey — page <span class="pageNumber"></span> of <span class="totalPages"></span></div>',
 margin:{top:'17mm',bottom:'18mm',left:'17mm',right:'17mm'}});
await b.close();})();
