import fs from "node:fs";

import path from "node:path";

import { fileURLToPath } from "node:url";

// Rebuild every committed SVG from the checked-in real benchmark JSON. No dependencies.

function esc(v){return String(v).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}

function svgFlow(){
const out=[];
const add=(s)=>out.push(s);
add(`<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="650" viewBox="0 0 1320 650" role="img" aria-labelledby="title desc"><title id="title">KUR-CF algorithm pipeline</title><desc id="desc">Train-only ratings and genre features form user attribute profiles; variance-weighted K-means clusters users; same-cluster users are scored with Pearson and Jaccard; nearest neighbors produce recommendations.</desc>`);
add(`<rect width="1320" height="650" fill="#ffffff"/><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#7b8da4"/></marker></defs>`);
add(`<text x="54" y="59" font-family="Arial,sans-serif" font-weight="700" font-size="30" fill="#15263c">KUR-CF | Algorithm Pipeline</text><text x="55" y="90" font-family="Arial,sans-serif" font-size="16" fill="#64748b">From training interactions to cluster-restricted collaborative filtering</text>`);
function card(x,y,w,h,tag,title,sub,stroke="#d8e2ed",fill="#f8fafc"){
add(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="16" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`);
add(`<text x="${x+20}" y="${y+28}" font-family="Arial,sans-serif" font-size="12" font-weight="700" letter-spacing="1" fill="#61758c">${esc(tag)}</text>`);
add(`<text x="${x+20}" y="${y+60}" font-family="Arial,sans-serif" font-size="19" font-weight="700" fill="#172b46">${esc(title)}</text>`);
add(`<text x="${x+20}" y="${y+85}" font-family="Arial,sans-serif" font-size="13" fill="#64748b">${esc(sub)}</text>`);
}
function arrow(d){add(`<path d="${d}" fill="none" stroke="#8293a9" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arr)"/>`);}
card(55,140,235,105,"INPUT 01","Ratings matrix X","Users × movies", "#d5e2f3","#f3f8ff");
card(55,285,235,105,"INPUT 02","Genre matrix Y","Movies × attributes","#d5e2f3","#f3f8ff");
card(365,215,245,116,"PAPER · EQ. 3","User profiles Z","Matrix product Z = X · Y","#b8d9d2","#f0faf7");
card(687,215,245,116,"PAPER · EQ. 4–5","Variance weights","w = Var / total variance","#b8d9d2","#f0faf7");
card(1010,215,255,116,"PAPER · EQ. 6–8","Weighted K-means","k = 7 / 6 in smoke runs","#b8d9d2","#f0faf7");
card(1010,431,255,117,"CLUSTER FILTER","Same-cluster peers","Restrict candidate users","#d5e2f3","#f3f8ff");
card(687,431,245,117,"PAPER · EQ. 9–10","Hybrid similarity","0.3 Pearson + 0.7 Jaccard","#b8d9d2","#f0faf7");
card(365,431,245,117,"NEIGHBORHOOD","Top-N similar users","N = 20, positive similarity","#d5e2f3","#f3f8ff");
card(55,431,235,117,"REPRODUCTION CHOICE","Rating prediction","Mean-centered + Top-K","#f4d5aa","#fff8ec");
arrow("M290 191 L322 191 L322 256 L360 256");
arrow("M290 338 L322 338 L322 291 L360 291");
arrow("M610 274 L682 274");
arrow("M932 274 L1005 274");
arrow("M1138 331 L1138 426");
arrow("M1009 490 L938 490");
arrow("M686 490 L615 490");
arrow("M364 490 L296 490");
add(`<line x1="54" y1="590" x2="1265" y2="590" stroke="#e5eaf0"/><text x="55" y="617" font-family="Arial,sans-serif" font-size="14" fill="#64748b">Green: paper-described operations · Orange: explicit prediction convention · All user statistics fitted from TRAIN ratings only.</text>`);
add("</svg>");
return out.join("\n")+"\n";
}

function svgChart(obj,display){
const config=obj.configuration, results=obj.results;
const colors={"kurcf":"#2463a7","usercf":"#0f9c8a","cluster-pearson":"#d68a29"};
const names={"kurcf":"KUR-CF","usercf":"UserCF","cluster-pearson":"Cluster-Pearson"};
const methods=["kurcf","usercf","cluster-pearson"];
const metrics=[
{key:"mae",title:"MAE",direction:"LOWER IS BETTER",format:v=>v.toFixed(4)},
{key:"rmse",title:"RMSE",direction:"LOWER IS BETTER",format:v=>v.toFixed(4)},
{key:"precision_at_k",title:"Precision@10",direction:"HIGHER IS BETTER",format:v=>(v*100).toFixed(2)+"%",factor:100},
{key:"recall_at_k",title:"Recall@10",direction:"HIGHER IS BETTER",format:v=>(v*100).toFixed(2)+"%",factor:100}
];
const out=[];
const add=s=>out.push(s);
add(`<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="865" viewBox="0 0 1280 865" role="img" aria-labelledby="title desc"><title id="title">${esc(display)} benchmark results</title><desc id="desc">Four bar charts of measured MAE, RMSE, Precision at 10 and Recall at 10 for KUR-CF, UserCF and Cluster-Pearson, from independently sampled MovieLens test data.</desc>`);
add(`<rect width="1280" height="865" fill="#ffffff"/><text x="55" y="64" font-family="Arial,sans-serif" font-weight="700" font-size="32" fill="#16283f">${esc(display)} | Reproduction Results</text>`);
add(`<text x="55" y="98" font-family="Arial,sans-serif" font-size="17" fill="#61758c">Train: ${config.train_interactions.toLocaleString("en-US")} · Test: ${config.test_interactions.toLocaleString("en-US")} · Seed: ${config.seed} · Clusters: ${config.clusters} · Neighbors: ${config.neighbors}</text>`);
add(`<text x="55" y="127" font-family="Arial,sans-serif" font-size="15" fill="#64748b">Rating cases: ${results.kurcf.n} · Ranking users: ${results.kurcf.users} · Top-K: ${results.kurcf.k} · Sampled smoke test, not the paper's original split</text>`);
methods.forEach((m,i)=>{
const x=55+i*200;
add(`<rect x="${x}" y="150" width="18" height="18" rx="4" fill="${colors[m]}"/><text x="${x+27}" y="165" font-family="Arial,sans-serif" font-size="15" fill="#334155">${names[m]}</text>`);
});
metrics.forEach((metric,index)=>{
const px=index%2===0?55:660,py=index<2?195:506,pw=565,ph=295;
const vals=methods.map(m=>results[m][metric.key]*(metric.factor||1));
const max=Math.max(...vals)*1.22 || 1;
const barX=px+172, maxWidth=275;
add(`<rect x="${px}" y="${py}" width="${pw}" height="${ph}" rx="18" fill="#f9fbfe" stroke="#dfe7f0" stroke-width="1.3"/>`);
add(`<text x="${px+25}" y="${py+43}" font-family="Arial,sans-serif" font-size="23" font-weight="700" fill="#1c334e">${metric.title}</text>`);
add(`<text x="${px+25}" y="${py+67}" font-family="Arial,sans-serif" font-size="12" font-weight="700" letter-spacing=".6" fill="#71839a">${metric.direction}</text>`);
methods.forEach((m,i)=>{
const y=py+103+i*55;
const v=vals[i],width=Math.max(0,(v/max)*maxWidth);
add(`<text x="${px+25}" y="${y+19}" font-family="Arial,sans-serif" font-size="14" fill="#42546c">${names[m]}</text>`);
add(`<rect x="${barX}" y="${y}" width="${maxWidth}" height="28" rx="6" fill="#e9eef5"/>`);
add(`<rect x="${barX}" y="${y}" width="${width.toFixed(3)}" height="28" rx="6" fill="${colors[m]}"/>`);
add(`<text x="${(barX+width+9).toFixed(2)}" y="${y+20}" font-family="Arial,sans-serif" font-size="14" font-weight="700" fill="#23354c">${metric.format(results[m][metric.key])}</text>`);
});
add(`<text x="${barX}" y="${py+281}" font-family="Arial,sans-serif" font-size="12" fill="#8190a2">0</text>`);
});
add(`<line x1="55" y1="826" x2="1225" y2="826" stroke="#e4e9f1"/>`);
add(`<text x="55" y="849" font-family="Arial,sans-serif" font-size="13" fill="#64748b">Source: committed benchmarks/*.json · Bars start at zero · Rankings have substantial uncertainty at this sample size.</text>`);
add("</svg>");
return out.join("\n")+"\n";
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const out = path.join(root, "assets", "figures");

fs.mkdirSync(out, { recursive: true });

const read = (name) => JSON.parse(fs.readFileSync(path.join(root, "benchmarks", name), "utf8"));

const figures = [["kurcf-pipeline.svg", svgFlow()], ["movielens-100k-results.svg", svgChart(read("ml100k-smoke.json"), "MovieLens 100K")], ["movielens-1m-results.svg", svgChart(read("ml1m-smoke.json"), "MovieLens 1M")]];

for (const [name, content] of figures) { fs.writeFileSync(path.join(out, name), content, "utf8"); console.log("Generated:", name); }
