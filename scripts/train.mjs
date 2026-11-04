import fs from 'node:fs';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {loadStylique} from './stylique-source.mjs';
const C=createRequire(import.meta.url)('../web/core.js');
const {splits,info:datasetInfo}=loadStylique();
const productId=i=>i.id;
function eligible(o){return o.items;}
const catalog=new Map(),outfits=[];let skippedDuplicates=0;
const seenOutfit=new Set();
for(const o of splits.train){const male=eligible(o);for(const i of male){const id=productId(i);if(!catalog.has(id))catalog.set(id,{...i,sourceSet:String(o.set_id),sourceIndex:i.index});}const ids=[...new Set(male.map(productId))];if(ids.length<2)continue;const key=[...ids].sort().join(',');if(seenOutfit.has(key)){skippedDuplicates++;continue;}seenOutfit.add(key);outfits.push({id:String(o.set_id),name:o.name,items:ids});}
const items=[...catalog.values()],df={};if(items.length<8||outfits.length<8)throw Error('Không đủ dữ liệu nam để huấn luyện 8 cụm.');for(const item of items)for(const t of new Set(C.tokens(item.document)))df[t]=(df[t]||0)+1;
const idf=Object.fromEntries(Object.entries(df).filter(([,n])=>n>=2).map(([t,n])=>[t,Math.log((1+items.length)/(1+n))+1]));
for(const item of items)item.vector=C.vector(item.document,idf);
for(const o of outfits)o.vector=C.aggregate(o.items.map(id=>catalog.get(id).vector));
console.log(`Lọc train: ${items.length} sản phẩm nam, ${outfits.length} bộ có ít nhất hai món nam.`);
const clustering=C.kmeans(outfits.map(o=>o.vector),8,30);outfits.forEach((o,i)=>o.cluster=clustering.labels[i]);
const clusters=clustering.centers.map((center,id)=>({id,count:outfits.filter(o=>o.cluster===id).length,keywords:Object.entries(center).sort((a,b)=>b[1]-a[1]).slice(0,6).map(([t])=>t)}));
const model={ready:true,source:datasetInfo.source,dataset:datasetInfo,items,outfits,idf,clusters,centers:clustering.centers};
// Item-disjointness is not claimed: original splits are at outfit level.
// Exclude contexts that reproduce a train item-set before evaluating one held-out item.
const trainSets=new Set(outfits.map(o=>[...o.items].sort().join(',')));
function evaluation(split,k){
  const encode=i=>({id:productId(i),role:i.role,vector:C.vector(i.document,idf)});
  const pool=[...new Map(splits[split].flatMap(eligible).map(i=>[productId(i),encode(i)])).values()];
  const hash=s=>crypto.createHash('sha256').update(s).digest('hex');
  let queries=0,hits=0,mrr=0,cosineHits=0,unseenTargets=0,exactSetsExcluded=0,trainItemOverlapExcludedOutfits=0;
  for(const o of splits[split]){
    const unique=[...new Map(eligible(o).map(i=>[productId(i),encode(i)])).values()];
    if(unique.length<2)continue;
    if(unique.some(i=>catalog.has(i.id))){trainItemOverlapExcludedOutfits++;continue;}
    const ids=new Set(unique.map(i=>i.id));
    if(trainSets.has([...ids].sort().join(','))){exactSetsExcluded++;continue;}
    for(const target of unique){
      if(!['top','bottom','shoes','outerwear','suit'].includes(target.role))continue;
      const context=unique.filter(i=>i.id!==target.id);
      const negatives=pool.filter(i=>i.role===target.role&&!ids.has(i.id)).map(i=>({item:i,key:hash(o.set_id+':'+target.id+':'+i.id)})).sort((a,b)=>a.key.localeCompare(b.key)).slice(0,3).map(x=>x.item);
      if(negatives.length<3)continue;
      const candidates=[target,...negatives],q=C.queryVector(context,'',model),neighbors=C.knn(q,outfits,k);
      const ranking=C.rankCandidates(q,candidates,neighbors,model),rank=ranking.findIndex(x=>x.item.id===target.id);
      queries++;if(!catalog.has(target.id))unseenTargets++;if(rank===0)hits++;mrr+=1/(rank+1);
      const simple=candidates.map(item=>({item,score:C.cosine(q,item.vector)})).sort((a,b)=>b.score-a.score||a.item.id.localeCompare(b.item.id));if(simple[0].item.id===target.id)cosineHits++;
    }
  }
  return {split,k,task:'Custom male 4-choice leave-one-item-out, three same-role negatives from the same held-out split',queries,hits,accuracy:queries?hits/queries:null,mrr:queries?mrr/queries:null,randomBaseline:.25,cosineBaselineAccuracy:queries?cosineHits/queries:null,unseenTargets,exactSetsExcluded,trainItemOverlapExcludedOutfits};
}
const validation=[5,10,20].map(k=>evaluation('valid',k));const best=validation.slice().sort((a,b)=>{const accuracy=b.accuracy-a.accuracy,mrr=b.mrr-a.mrr;return Math.abs(accuracy)>1e-12?accuracy:Math.abs(mrr)>1e-12?mrr:a.k-b.k;})[0];model.defaultK=best.k;
const test=evaluation('test',best.k);
const stats={...datasetInfo,rawOutfits:Object.fromEntries(Object.entries(splits).map(([s,rows])=>[s,rows.length])),items:items.length,outfits:outfits.length,roles:Object.fromEntries(Object.keys(C.roles).map(r=>[r,items.filter(i=>i.role===r).length])),clusters,iterations:clustering.iterations,vocabulary:Object.keys(idf).length,skippedDuplicateTrainSets:skippedDuplicates,validation,test,imagesIncluded:false,filter:'Explicit male category path / description gender / unambiguously male category; explicit female always excluded'};
fs.writeFileSync('web/catalog.js','window.CATALOG = '+JSON.stringify(model)+';\n');
fs.writeFileSync('web/stats.js','window.STATS = '+JSON.stringify(stats)+';\n');
fs.writeFileSync('data/model-report.json',JSON.stringify(stats,null,2)+'\n');
console.log(JSON.stringify({items:stats.items,outfits:stats.outfits,roles:stats.roles,validation,test},null,2));
