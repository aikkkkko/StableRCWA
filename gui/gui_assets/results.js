"use strict";
// Display reads stored complex fields. No component switch launches a solve.
let viewRequest=0;
function fileUrl(name){return `/api/jobs/${lastJob.id}/files/${name}`;}
function metric(label,value,detail=''){return `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="detail">${esc(detail)}</div></div>`;}
function resultDuration(value){return value==null?'—':Number(value)<1?`${fmt(Number(value)*1000,3)} ms`:`${fmt(value,3)} s`;}
// Which solver produced efficiencies / fields / phase, plus optional explicit verification.
function backendLine(b){
  const who=k=>k==='cpp'?'C++（StableRCWA）':'Python',sci=v=>v==null?'—':Number(v).toExponential(1);
  let s=`衍射效率：${who(b.efficiency)} · 场图：${who(b.fields)}${b.phase?` · 相位：${who(b.phase)}`:''}`;
  if(b.max_efficiency_difference!=null)s+=` · 两个求解器各级效率最大相差 ${sci(b.max_efficiency_difference)}${b.max_jones_difference==null?'':` · Jones 矩阵最大相差 ${sci(b.max_jones_difference)}`}`;
  if(b.efficiency==='cpp')s+=` · C++ 用时 ${resultDuration(b.cpp_seconds)}${b.python_solve_ran||b.python_solve_seconds>0?`（Python 核对 ${resultDuration(b.python_solve_seconds)}）`:'（一次求解）'}`;
  if(b.note)s+=` · ${b.efficiency==='cpp'?'':'未用 C++：'}${b.note}`;
  // Conical only: C++ found two modes of a grating layer nearly coincident (degenerate_mode_warning).
  if(b.fields==='cpp'&&b.n_schur>0)return `${s} · C++ 对 ${b.n_schur} 个层使用 Schur 传播，场图保留完整矩阵耦合`;
  if(b.fields==='cpp'&&b.cpp_zero_modes>0)s+=` · C++ 场图保留 ${b.cpp_zero_modes} 个精确零根的线性场项`;
  if(b.fields==='python'&&b.python_n_schur>0)return `${s} · Python 对 ${b.python_n_schur} 个层使用 Schur 传播，场图保留完整矩阵耦合`;
  if(b.python_zero_modes>0)s+=` · Python 场图保留 ${b.python_zero_modes} 个精确零根的线性场项`;
  if(b.degenerate_mode_warning&&b.efficiency==='cpp'&&b.n_schur>0)return `${s} · C++ 已对 ${b.n_schur} 个退化层使用 Schur 传播；该场图的 Python 结果未应用此后备算法`;
  return b.degenerate_mode_warning?`${s} · 注意：在这个波长和角度下，某个光栅层有两个模式几乎重合，结果的精度没有保证（Python 求解器同样如此）；请把波长或某个角度稍微挪动一点再算`:s;
}
function historyLabel(r){const i=r.incident;return `${r.id.slice(4,8)} ${r.id.slice(9,11)}:${r.id.slice(11,13)} · ${r.name} · ${i.polarization} / M${i.M}${r.mode==='periodic2d'?'×'+i.My:''}`;}
function renderHistory(records){
  const selectedId=lastJob?.id||$('resultHistory').value;
  $('resultHistory').innerHTML='<option value="">选择已保存的计算…</option>'+records.map(r=>`<option value="${esc(r.id)}">${esc(historyLabel(r))}</option>`).join('');
  if(records.some(r=>r.id===selectedId))$('resultHistory').value=selectedId;
}
async function refreshHistory(){try{renderHistory(await api('/api/history'));}catch(e){error(e);}}
async function openSavedResult(id,restoreModel=false){
  if(!id)return;
  try{
    clearError();const job=await api('/api/jobs/'+encodeURIComponent(id));
    if(job.status!=='done')throw Error('该记录尚无完整结果');
    if(restoreModel){
      config=clone(job.config);$('structureView').value=config.profile_geometry?'profile':'surface';selected=0;layerDirty=false;$('preset').value='custom_current';
      syncInputs();renderEditor();changed();await previewNow();runRevision=revision;
    }
    showResult(job,{historical:!restoreModel});
  }catch(e){error(e);}
}
async function adoptResultModel(){
  if(!lastJob||busy)return;
  config=clone(lastJob.config);selected=0;layerDirty=false;$('preset').value='custom_current';
  syncInputs();renderEditor();changed();
  if(await previewNow()){runRevision=revision;$('staleNotice').hidden=true;switchTab('structure');}
}
async function refreshResultView(){
  if(!lastJob)return;
  const sequence=++viewRequest,id=lastJob.id;
  const component=$('fieldComponent').value,aggregate=component==='E2'||component==='H2';
  if(aggregate)$('fieldQuantity').value='abs2';
  $('fieldQuantity').disabled=aggregate;
  const quantity=$('fieldQuantity').value,source=$('fieldSource').value,palette=$('fieldPalette').value;
  $('viewStatus').textContent='正在从已保存复场绘图…';
  $('imageDownload').removeAttribute('href');
  try{
    const meta=await api('/api/plot',{id,component,quantity,source,palette});
    if(sequence!==viewRequest||lastJob.id!==id)return;
    $('resultImage').onerror=()=>{if(sequence===viewRequest)$('viewStatus').textContent='图像载入失败，请重新选择分量。';};
    $('resultImage').src=fileUrl(meta.filename);$('resultImage').alt=meta.title;
    $('imageDownload').href=fileUrl(meta.filename);$('imageDownload').download=meta.filename;
    $('componentNote').textContent=meta.note;
    $('componentNote').classList.toggle('zero-note',meta.zero_component);
    const range=meta.minimum===null?'此显示量没有有限数值':`范围 ${Number(meta.minimum).toPrecision(5)} ～ ${Number(meta.maximum).toPrecision(5)}`;
    const url=new URL(window.location.href);if(palette==='auto')url.searchParams.delete('palette');else url.searchParams.set('palette',palette);window.history.replaceState(null,'',url);
    $('viewStatus').textContent=`${meta.colorbar} · ${meta.colormap||'auto'} · ${range} · ${meta.cache_hit?'已缓存图像':`绘图 ${fmt(meta.request_seconds??0,3)} s`} · 无需重新求解`;
  }catch(e){if(sequence===viewRequest){$('viewStatus').textContent='绘图未完成';error(e);}}
}
function showResult(job,{historical=false}={}){
  lastJob=job;if(!busy&&!activeJob)$('runStatus').textContent=job.stage||'计算完成';const m=job.metrics,vector=m.mode==='conical'||m.mode==='periodic2d';
  $('emptyResult').hidden=true;$('resultContent').hidden=false;
  $('staleNotice').hidden=!historical&&revision===runRevision;
  $('staleNotice').textContent=historical?'当前显示已保存的计算结果。若要继续编辑这个结构，可点击“载入此模型”。':'参数已修改。下方保留的是上一次计算结果，请重新计算以更新。';
  $('resultName').textContent=m.name;AdaptivePMLControls.renderResult(m.adaptive_pml);for(const id of ['downloadAdaptive','downloadRequested'])$(id).hidden=!m.adaptive_pml;if(m.adaptive_pml){$('downloadAdaptive').href=fileUrl('adaptive_pml.json');$('downloadRequested').href=fileUrl('requested_model.json');}
  $('resultConditions').textContent=`${m.mode==='pml'?'横向 PML':m.mode==='periodic2d'?'二维周期 RCWA':m.mode==='conical'?'周期圆锥 RCWA':'周期 RCWA'} · ${m.incident.polarization} · λ ${fmt(m.incident.wavelength)} μm · θ ${fmt(m.incident.angle_deg)}°${vector?` · φ ${fmt(m.incident.phi_deg,6)}° · ψ ${fmt(m.incident.psi_deg)}° · δ ${fmt(m.incident.delta_deg)}°${m.mode==='conical'?' · y=0':` · ${m.field_plane} / 固定${m.field_plane==='xz'?'y':'x'}=${fmt(m.field_at)} μm`}`:''}${m.mode==='pml'&&m.incident.polarization==='jones'?` · ψ ${fmt(m.incident.psi_deg)}° · δ ${fmt(m.incident.delta_deg)}° · TE+TM 严格叠加`:''} · M ${m.incident.M}${m.mode==='periodic2d'?' × '+m.incident.My:''}`;
  const times=m.timings;$('resultStages').textContent=times?`场重建 ${resultDuration(times.field_reconstruction_seconds)}${m.backend?.cpp_phase_seconds!=null?` · 相位 ${resultDuration(m.backend.cpp_phase_seconds)}`:''} · 数据处理 ${resultDuration(times.postprocess_seconds)} · 保存 ${resultDuration(times.save_seconds)} · 工作进程加载 ${resultDuration(times.worker_import_seconds)}${times.blas_threads?` · BLAS线程 ${times.blas_threads}`:''}`:'原记录未分项计时';
  $('resultBackend').hidden=!m.backend;$('resultBackend').textContent=m.backend?backendLine(m.backend):'';
  $('resultTiming').textContent=`${m.adaptive_pml?'选参与验证':m.backend?.fields==='cpp'?'C++求解':'原求解'} ${resultDuration(m.solve_seconds)}`;
  const fs=m.fourier_layers||[],ff=fs.filter(l=>l.method==='fft'),an=fs.filter(l=>l.method==='analytic');
  $('resultFourier').textContent=ff.length?`材料系数：${ff.length} 层FFT · N=${[...new Set(ff.map(l=>l.fft_samples))].join('/')}，${an.length} 层解析积分`:(an.length?`材料系数：${an.length} 层解析积分`:'原结果 / 均匀层解析模式');
  if(m.profile_geometry){const g=m.profile_geometry;$('resultFourier').textContent+=` · 斜壁 ${g.slices} 片 · 目标上/下底 ${fmt(g.top_width*1000,3)}/${fmt(g.bottom_width*1000,3)} nm`;}
  let html=metric('总电场E²采样峰值',fmt(m.sampled_E2_peak,4),'相对入射 |E|²')+metric('E²峰值位置',`${m.field_plane==='yz'?'y':'x'} ${fmt(m.field_plane==='yz'?m.peak_y:m.peak_x,3)}`,`z ${fmt(m.peak_z,3)} μm`);
  if(m.mode!=='pml')html+=metric('总反射效率',(100*m.R_total).toFixed(3)+'%')+metric('总透射效率',(100*m.T_total).toFixed(3)+'%');
  else html+=metric('结构层数',m.layer_count,'一维横向结构')+metric(m.timings?'计算与保存':'原计算与出图',fmt(m.total_seconds,2)+' s');
  $('metrics').innerHTML=html;
  $('fieldSource').value='total';$('contrastOption').disabled=m.mode!=='pml';
  $('fieldComponent').value='E2';$('fieldQuantity').value='abs2';
  $('orderSection').hidden=m.mode==='pml';
  document.querySelectorAll('.conical-order').forEach(e=>e.hidden=!vector);
  $('orderLabel').textContent=m.mode==='periodic2d'?'级次 (m,n)':'级次 m（程序约定）';
  $('orderRows').innerHTML=m.orders.map(o=>`<tr><td>${m.mode==='periodic2d'?`(${o.m}, ${o.n})`:o.m}</td><td>${(100*o.R).toFixed(5)}%</td><td>${(100*o.T).toFixed(5)}%</td>${vector?`<td>${(100*o.R_s).toFixed(5)}% / ${(100*o.R_p).toFixed(5)}%</td><td>${(100*o.T_s).toFixed(5)}% / ${(100*o.T_p).toFixed(5)}%</td>`:''}</tr>`).join('');
  $('resultNote').textContent=m.mode==='pml'?'电场分量归一到 |E_inc|，磁场分量归一到 |H_inc|。PML灰区不作为物理总场。':`电场/磁场分别归一到各自入射幅值。功率差额 1−R−T = ${fmt(100*m.absorption,6)}%。${vector?' 各级s/p分量按该衍射级的出射面定义。':''}${m.mode==='conical'?'六分量数据含复Jones矩阵Jr/Jt。':''}`;
  $('downloadFields').href=fileUrl('fields_six.npz');$('downloadConfig').href=fileUrl('model.json');$('downloadMetrics').href=fileUrl('metrics.json');
  $('resultFolder').textContent='已保存：'+job.folder;$('adoptResultModel').disabled=false;
  const url=new URL(window.location.href);url.searchParams.set('result',job.id);window.history.replaceState(null,'',url);
  refreshHistory();refreshResultView();switchTab('result');
}
function initResultControls(history){
  renderHistory(history||[]);const palette=new URLSearchParams(window.location.search).get('palette');if([...$('fieldPalette').options].some(o=>o.value===palette))$('fieldPalette').value=palette;
  $('resultHistory').addEventListener('change',()=>openSavedResult($('resultHistory').value));
  $('adoptResultModel').addEventListener('click',adoptResultModel);
  for(const id of ['fieldComponent','fieldQuantity','fieldSource','fieldPalette'])$(id).addEventListener('change',refreshResultView);
}
