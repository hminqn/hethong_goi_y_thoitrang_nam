import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawn} from 'node:child_process';
import {pipeline} from 'node:stream/promises';
import {Writable} from 'node:stream';
const root=path.resolve('datasets/Stylique-Polyvore');
const metadata=JSON.parse(fs.readFileSync(path.join(root,'.download/repository.json'),'utf8'));
const revision=metadata.sha;
if(!/^[a-f0-9]{40}$/.test(revision))throw Error('Invalid revision.');
const files=metadata.siblings;
for(const f of files){const target=path.resolve(root,f.rfilename);if(!target.startsWith(root+path.sep)||f.rfilename.includes('..')||!Number.isSafeInteger(f.size))throw Error('Invalid file path or size.');}
const reportPath=path.join(root,'.download/download-report.json');
const report={repository:'Stylique/Polyvore',revision,expectedFiles:files.length,expectedBytes:files.reduce((s,f)=>s+f.size,0),startedAt:new Date().toISOString(),files:{},status:'downloading'};
function save(){fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+'\n');}
save();
async function checksum(file,algorithm,gitBlob=false){const h=crypto.createHash(algorithm);if(gitBlob)h.update(`blob ${fs.statSync(file).size}\0`);await pipeline(fs.createReadStream(file),new Writable({write(chunk,enc,cb){h.update(chunk);cb();}}));return h.digest('hex');}
function curl(url,dest){return new Promise(resolve=>{const p=spawn('curl.exe',['--location','--fail','--silent','--show-error','--continue-at','-','--connect-timeout','30','--max-time','900','--speed-time','90','--speed-limit','1024','--output',dest,url],{stdio:['ignore','ignore','pipe'],windowsHide:true});let error='';p.stderr.on('data',b=>{error+=b.toString();if(error.length>3000)error=error.slice(-3000);});p.on('error',e=>resolve({code:-1,error:e.message}));p.on('close',code=>resolve({code,error:error.trim()}));});}
async function verify(f,target){if(!fs.existsSync(target)||fs.statSync(target).size!==f.size)return null;if(f.lfs?.sha256){const sha256=await checksum(target,'sha256');if(sha256!==f.lfs.sha256)throw Error('SHA-256 mismatch: '+f.rfilename);return {bytes:f.size,sha256,verified:'LFS SHA-256'};}const blob=await checksum(target,'sha1',true);if(f.blobId&&blob!==f.blobId)throw Error('Git SHA-1 mismatch: '+f.rfilename);return {bytes:f.size,gitBlobSHA1:blob,verified:f.blobId?'Git blob SHA-1':'size'};}
async function download(f){const target=path.resolve(root,f.rfilename),part=target+'.part';fs.mkdirSync(path.dirname(target),{recursive:true});const existing=await verify(f,target);if(existing){report.files[f.rfilename]=existing;save();console.log('Verified existing: '+f.rfilename);return;}if(fs.existsSync(target))throw Error('Existing final file has wrong size: '+f.rfilename);const url=`https://huggingface.co/datasets/Stylique/Polyvore/resolve/${revision}/${f.rfilename.split('/').map(encodeURIComponent).join('/')}?download=true`;
for(let attempt=1;attempt<=8;attempt++){const size=fs.existsSync(part)?fs.statSync(part).size:0;if(size>f.size)throw Error('Oversized partial file: '+f.rfilename);if(size===f.size)break;console.log(`Downloading ${f.rfilename} (${size}/${f.size} bytes), attempt ${attempt}`);const result=await curl(url,part);if(result.code===0&&fs.existsSync(part)&&fs.statSync(part).size===f.size)break;if(result.code===33||result.code===36)throw Error('Server refused resume: '+f.rfilename);console.log(`Retry ${f.rfilename}: ${result.error||'incomplete download'}`);}
const verified=await verify(f,part);if(!verified)throw Error('Incomplete file after retries: '+f.rfilename);fs.renameSync(part,target);report.files[f.rfilename]=verified;save();console.log(`Complete & verified: ${f.rfilename}`);}
let cursor=0;const errors=[];
// Small files complete first; three concurrent transfers bound network and disk use.
const queue=[...files].sort((a,b)=>a.size-b.size);
const timer=setInterval(()=>{const complete=Object.keys(report.files).length;let bytes=0;for(const f of files){const target=path.resolve(root,f.rfilename);for(const p of [target,target+'.part'])if(fs.existsSync(p)){bytes+=fs.statSync(p).size;break;}}console.log(`PROGRESS ${complete}/${files.length} files; ${(bytes/1e9).toFixed(2)}/${(report.expectedBytes/1e9).toFixed(2)} GB (${(100*bytes/report.expectedBytes).toFixed(1)}%)`);},20000);
try{await Promise.all(Array.from({length:3},async()=>{while(cursor<queue.length){const f=queue[cursor++];try{await download(f);}catch(error){errors.push({file:f.rfilename,error:error.message});console.log('ERROR '+error.message);}}}));}finally{clearInterval(timer);}
report.finishedAt=new Date().toISOString();report.status=errors.length?'incomplete':'complete';report.errors=errors;save();
console.log(JSON.stringify({status:report.status,verifiedFiles:Object.keys(report.files).length,expectedFiles:files.length,bytes:report.expectedBytes,root,errors},null,2));if(errors.length)process.exitCode=1;
