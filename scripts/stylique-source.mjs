import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
export const DATASET_SOURCE='https://huggingface.co/datasets/Stylique/Polyvore';
export function categoryMap(csv){const map=new Map();for(const line of csv.trim().split(/\r?\n/)){const m=line.match(/^(\d+),([^,]*),(.*)$/);if(!m)continue;const entry={fine:m[2].trim(),semantic:m[3].trim()};if(!map.has(m[1]))map.set(m[1],[]);map.get(m[1]).push(entry);}return map;}
export function maleItem(id,meta,categories){
  const catId=String(meta.category_id||''),options=categories.get(catId)||[],trail=Array.isArray(meta.catgeories)?meta.catgeories.filter(x=>typeof x==='string'):Array.isArray(meta.categories)?meta.categories:[];
  const description=String(meta.description||''),name=String(meta.title||meta.url_name||'').trim();
  const female=trail.some(x=>/\bwomen(?:['’]s)?\b|\bfemale\b/i.test(x))||/\bgender\s*:\s*female\b/i.test(description);
  if(female)return {excluded:'explicit_female'};
  let evidence=null;
  if(trail.some(x=>/\bmen(?:['’]s)?\b|\bmenswear\b/i.test(x)))evidence='category_path';
  else if(/\bgender\s*:\s*male\b/i.test(description))evidence='description_gender';
  else if(options.length&&options.every(x=>/^male\b/i.test(x.fine)))evidence='male_only_category';
  if(!evidence)return {excluded:'no_male_evidence'};
  const semantic=String(meta.semantic_category||options[0]?.semantic||''),fine=options.map(x=>x.fine).filter(Boolean).join(' / ');
  let role={tops:'top',bottoms:'bottom',shoes:'shoes',outerwear:'outerwear',accessories:'accessory',bags:'accessory',hats:'accessory',sunglasses:'accessory',jewellery:'accessory',jewelry:'accessory'}[semantic];
  if(semantic==='all-body'&&/suit/i.test(name+' '+fine))role='suit';
  if(/boxer|underwear|swim|sleep|groom|fragrance|shav|skin|hair|deodorant/i.test(fine))return {excluded:'unsupported_category'};
  if(!role||!name)return {excluded:'unsupported_category'};
  return {item:{id:String(id),name,category:trail.slice(0,3).join(' / ')||fine||semantic,categoryId:catId,role,price:null,genderEvidence:evidence,semanticCategory:semantic,document:name+' '+fine+' '+semantic}};
}
export function loadStylique(root='datasets/Stylique-Polyvore'){
  const metadataFile=path.join(root,'polyvore_item_metadata.json');
  if(!fs.existsSync(metadataFile))throw Error('Metadata Stylique chưa tải xong: cần polyvore_item_metadata.json đầy đủ. Dự án không sử dụng dataset cũ.');
  const bytes=fs.readFileSync(metadataFile),digest=crypto.createHash('sha256').update(bytes).digest('hex');
  const repository=JSON.parse(fs.readFileSync(path.join(root,'.download/repository.json'),'utf8'));
  const expected=repository.siblings.find(f=>f.rfilename==='polyvore_item_metadata.json');
  if(bytes.length!==expected.size||digest!==expected.lfs.sha256)throw Error('Metadata mới chưa qua kiểm tra kích thước/SHA-256.');
  const metadata=JSON.parse(bytes),categories=categoryMap(fs.readFileSync(path.join(root,'categories.csv'),'utf8')),titles=JSON.parse(fs.readFileSync(path.join(root,'polyvore_outfit_titles.json'),'utf8'));
  const records=new Map(),filterCounts={};
  for(const [id,meta]of Object.entries(metadata)){const result=maleItem(id,meta,categories);if(result.item)records.set(id,result.item);else filterCounts[result.excluded]=(filterCounts[result.excluded]||0)+1;}
  const splits=Object.fromEntries(['train','valid','test'].map(split=>{const raw=JSON.parse(fs.readFileSync(path.join(root,'disjoint',split+'.json'),'utf8'));return [split,raw.map(o=>({set_id:String(o.set_id),name:titles[o.set_id]?.title||titles[o.set_id]?.url_name||'Bộ '+o.set_id,items:o.items.filter(i=>records.has(String(i.item_id))).map(i=>({...records.get(String(i.item_id)),index:i.index}))}))];}));
  const ids=Object.fromEntries(Object.entries(splits).map(([s,rows])=>[s,new Set(rows.flatMap(o=>o.items.map(i=>i.id)))]));
  const maleItemOverlap={trainValid:[...ids.valid].filter(id=>ids.train.has(id)).length,trainTest:[...ids.test].filter(id=>ids.train.has(id)).length};
  return {splits,info:{source:DATASET_SOURCE,revision:repository.sha,scheme:'disjoint',metadataSHA256:digest,totalMetadataItems:Object.keys(metadata).length,maleSupportedItems:records.size,filterCounts,maleItemOverlap}};
}
