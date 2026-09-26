// Dump the membrane app's ALIGN enumeration (the pure section, evaluated verbatim) for
// align_reference_MIRROR.py.  Run from this folder:  node align_dump_MIRROR.js
const fs=require('fs'), path=require('path');
const html=fs.readFileSync(path.join(__dirname,'..','wave_membrane_MIRROR.html'),'utf8');
const src=html.split('/* ALIGN-BEGIN */')[1].split('/* ALIGN-END */')[0];
eval(src);
const out={};
for(const lat of ['sq','tri']){
  for(const sh of ['n3','n4','n5','n6','n8','n12'])
    out[`${lat}/${sh}`]=alAngles(alEdgeDirs0(sh),lat,0,90).map(e=>[e.rot,e.rows,e.diags]);
  for(let th2=40;th2<=180;th2++){ const th=th2/2;
    out[`${lat}/rhomb${th}`]=alAngles(alEdgeDirs0('rhomb',th*Math.PI/180),lat,0,90).map(e=>[e.rot,e.rows,e.diags]); }
}
fs.writeFileSync(path.join(__dirname,'align_js_MIRROR.json'), JSON.stringify(out));
console.log('dumped', Object.keys(out).length, 'configurations');
