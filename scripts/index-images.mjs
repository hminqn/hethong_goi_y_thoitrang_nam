import fs from 'node:fs';
import vm from 'node:vm';
const ctx={window:{}};vm.runInNewContext(fs.readFileSync('web/catalog.js','utf8'),ctx);
const index={};for(const item of ctx.window.CATALOG.items){for(const ext of ['jpg','png','webp']){const candidates=[`images/${item.id}.${ext}`,`images/${item.sourceSet}/${item.sourceIndex}.${ext}`];const match=candidates.find(p=>fs.existsSync('web/'+p));if(match){index[item.id]=match;break;}}}
fs.writeFileSync('web/image-index.js','window.PRODUCT_IMAGES = '+JSON.stringify(index)+';\n');
console.log(`Đã liên kết ${Object.keys(index).length}/${ctx.window.CATALOG.items.length} ảnh. Những món còn lại dùng minh họa loại đồ.`);
