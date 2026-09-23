(() => {
  const canvas = document.getElementById("timeline");
  if (!canvas || !window.CYBERWORLD_TIMELINE?.length) return;
  const data = window.CYBERWORLD_TIMELINE;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = 300;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  const pad = {l:42,r:18,t:20,b:34}, w=width-pad.l-pad.r, h=height-pad.t-pad.b;
  const x = i => pad.l + i/Math.max(1,data.length-1)*w;
  const y = v => pad.t + (1-v)*h;
  ctx.font="11px system-ui"; ctx.strokeStyle="#203541"; ctx.fillStyle="#8fa9ad"; ctx.lineWidth=1;
  for(let v=0;v<=1;v+=.25){ctx.beginPath();ctx.moveTo(pad.l,y(v));ctx.lineTo(width-pad.r,y(v));ctx.stroke();ctx.fillText(v.toFixed(2),5,y(v)+4)}
  ctx.fillStyle="rgba(255,98,111,.12)"; let open=false,start=0;
  data.forEach((d,i)=>{const flagged=d.risk_score>=window.CYBERWORLD_THRESHOLD;if(flagged&&!open){start=i;open=true}if(open&&(!flagged||i===data.length-1)){ctx.fillRect(x(start),pad.t,x(i)-x(start),h);open=false}});
  ctx.strokeStyle="#ff626f";ctx.setLineDash([6,5]);ctx.beginPath();ctx.moveTo(pad.l,y(window.CYBERWORLD_THRESHOLD));ctx.lineTo(width-pad.r,y(window.CYBERWORLD_THRESHOLD));ctx.stroke();ctx.setLineDash([]);
  const gradient=ctx.createLinearGradient(pad.l,0,width-pad.r,0);gradient.addColorStop(0,"#55e6b5");gradient.addColorStop(1,"#41bde8");ctx.strokeStyle=gradient;ctx.lineWidth=2;ctx.beginPath();data.forEach((d,i)=>i?ctx.lineTo(x(i),y(d.risk_score)):ctx.moveTo(x(i),y(d.risk_score)));ctx.stroke();
  const peak=data.reduce((a,d,i)=>d.risk_score>data[a].risk_score?i:a,0);ctx.fillStyle="#ff626f";ctx.beginPath();ctx.arc(x(peak),y(data[peak].risk_score),5,0,Math.PI*2);ctx.fill();
})();
