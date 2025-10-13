// serves index.html and proxies /ask to core
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json({ limit: '200kb' }));

const PORT = parseInt(process.env.PORT || '8099', 10);
const CORE_URL = process.env.CORE_URL || 'http://core:8000/v1/widget_text';
const CORE_HEALTH = process.env.CORE_HEALTH || 'http://core:8000/health';
const STATIC_DIR = process.env.STATIC_DIR || __dirname;

app.use((req,res,next)=>{res.set('Cache-Control','no-store, no-cache, must-revalidate, max-age=0');res.set('Pragma','no-cache');res.set('Expires','0');next();});

app.get('/health', async (_req,res)=>{
  const info={status:'ok',port:PORT,static_dir:STATIC_DIR,core:{url:CORE_URL,health:CORE_HEALTH}};
  try{const r=await axios.get(CORE_HEALTH,{timeout:2500});info.core.reachable=true;info.core.engine_version=r?.data?.engine_version??null;}
  catch(e){info.core.reachable=false;info.core.error=e.code||String(e);}
  res.json(info);
});
app.get('/version', async (_req,res)=>{
  try{const r=await axios.get(CORE_HEALTH,{timeout:2500});res.json({engine_version:r?.data?.engine_version??null});}
  catch(e){res.status(502).json({engine_version:null,error:e.code||String(e)});}
});
app.post('/ask', async (req,res)=>{
  try{
    const q=(req.body && typeof req.body.query==='string')?req.body.query.trim():'';
    if(!q) return res.status(400).json({error:'bad_request',detail:'Body must be {"query":"..."}'});
    const r=await axios.post(CORE_URL,{query:q},{timeout:10000,headers:{'Content-Type':'application/json'}});
    return res.status(200).json(r.data);
  }catch(err){
    const status=err?.response?.status||502; const detail=err?.response?.data||err?.code||String(err);
    return res.status(status).json({error:'proxy_failed',core:CORE_URL,detail});
  }
});
app.get('/', (_req,res)=>{
  const index=path.join(STATIC_DIR,'index.html');
  if(fs.existsSync(index)) return res.sendFile(index);
  res.status(404).send('index.html not found');
});
app.listen(PORT,'0.0.0.0',()=>console.log(`[widget] ${PORT} → CORE_URL=${CORE_URL} | static=${STATIC_DIR}`));
