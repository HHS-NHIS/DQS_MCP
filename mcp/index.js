const axios=require('axios'); const CORE_URL=process.env.CORE_URL||'http://core:8000/v1/widget_text';
function W(o){try{process.stdout.write(JSON.stringify(o));}catch(e){process.stdout.write('{"jsonrpc":"2.0","id":null,"error":{"code":-32001,"message":"Serialization error"}}');}}
function Err(id,code,msg,data){return{jsonrpc:"2.0",id,error:{code,message:msg,data}};} function Ok(id,result){return{jsonrpc:"2.0",id,result};}
async function handle(m){const{id,method,params={}}=m||{}; if(!method) return W(Err(null,-32600,"Invalid Request"));
 if(method==="tools/list") return W(Ok(id,{tools:[{name:"dqs.ask.core",description:"Ask core",inputSchema:{type:"object",properties:{query:{type:"string"}},required:["query"]}}]}));
 if(method==="tools/call"){ if(!params||params.name!=="dqs.ask.core") return W(Err(id,-32601,"Tool not found"));
  const q=(params.arguments&&typeof params.arguments.query==="string")?params.arguments.query:"";
  try{const r=await axios.post(CORE_URL,{query:q},{headers:{"Content-Type":"application/json"},timeout:15000}); return W(Ok(id,r.data));}
  catch(e){return W(Err(id,-32000,"Core request failed", e?.response?.data ?? String(e)));}}
 return W(Err(id,-32601,"Method not found"));
}
const chunks=[]; process.stdin.on('data',d=>chunks.push(d)); process.stdin.on('end',async()=>{let raw=Buffer.concat(chunks).toString('utf8').trim();let msg;try{msg=JSON.parse(raw);}catch(e){return W(Err(null,-32700,"Parse error",String(e)));} await handle(msg);}); process.stdin.resume();
