"use strict";

/* The host calls write/read when it synchronizes the complete model config. */
window.AdaptivePMLControls = (() => {
  const defaults = {
    enabled: false,
    target_tail: 0.001,
    verify: true,
    verification_quantity: 'main',
    verification_rtol: 0.01,
    max_solves: 8,
    max_M: 300,
    max_width_factor: 4,
    max_seconds: 300
  };
  let original = {};
  let busy = false;
  let root = null;

  const byId = id => root?.querySelector(`#${id}`);
  const asNumber = id => byId(id).valueAsNumber;
  const percent = value => Number(value) * 100;
  const displayNumber = (value, digits = 4) => {
    const number = Number(value);
    return Number.isFinite(number)
      ? number.toLocaleString('zh-CN', { maximumFractionDigits: digits, useGrouping: false })
      : '—';
  };
  const displayPercent = value => `${displayNumber(percent(value), 4)}%`;

  function updateEnabled() {
    if (!root) return;
    const enabled = byId('adaptivePmlEnabled').checked;
    const details = byId('adaptivePmlDetails');
    details.hidden = !enabled;
    details.querySelectorAll('input, select').forEach(control => { control.disabled = busy || !enabled; });
    byId('adaptivePmlEnabled').disabled = busy;
  }

  function init() {
    const target = document.getElementById('adaptivePmlControls');
    if (!target) return;
    if (root === target && target.querySelector('#adaptivePmlEnabled')) return;
    root = target;
    root.innerHTML = `
      <div class="apml-head">
        <label class="apml-switch" for="adaptivePmlEnabled">
          <input id="adaptivePmlEnabled" type="checkbox">
          <span>自动选择 PML 参数</span>
        </label>
      </div>
      <div id="adaptivePmlDetails" class="apml-details" hidden>
        <div class="apml-field">
          <label for="adaptivePmlTargetTail">边界对比场残余目标 <span>%</span></label>
          <input id="adaptivePmlTargetTail" type="number" step="any" min="0" value="0.1">
          <p class="apml-help">相对物理区对比场 RMS，属于吸收诊断，不是反射率或真实误差上界。</p>
        </div>
        <label class="apml-check" for="adaptivePmlVerify">
          <input id="adaptivePmlVerify" type="checkbox" checked>
          <span>另做物理区验证</span>
        </label>
        <p class="apml-help apml-verify-help">独立提高 M 与加厚 PML，检查物理区所选场量是否稳定。</p>
        <div class="apml-field">
          <label for="adaptivePmlVerificationQuantity">物理区验证场量</label>
          <select id="adaptivePmlVerificationQuantity">
            <option value="main">主复场（TE: Ey / TM: Hy）</option>
            <option value="electric">电场矢量（更严格）</option>
          </select>
        </div>
        <div class="apml-field">
          <label for="adaptivePmlVerificationRtol">物理区所选场量变化容许值 <span>%</span></label>
          <input id="adaptivePmlVerificationRtol" type="number" step="any" min="0" value="1">
        </div>
        <details class="apml-advanced">
          <summary>高级限制</summary>
          <div class="apml-grid">
            <div class="apml-field"><label for="adaptivePmlMaxM">最大 M</label><input id="adaptivePmlMaxM" type="number" step="1" min="1" value="300"></div>
            <div class="apml-field"><label for="adaptivePmlMaxWidthFactor">PML 最大厚度倍数</label><input id="adaptivePmlMaxWidthFactor" type="number" step="any" min="1" value="4"></div>
            <div class="apml-field"><label for="adaptivePmlMaxSolves">最多求解次数</label><input id="adaptivePmlMaxSolves" type="number" step="1" min="1" value="8"></div>
            <div class="apml-field"><label for="adaptivePmlMaxSeconds">计算预算 <span>秒，求解间检查</span></label><input id="adaptivePmlMaxSeconds" type="number" step="any" min="0" value="300"></div>
          </div>
        </details>
      </div>`;
    root.addEventListener('input', event => {
      if (!event.target.matches('input, select')) return;
      if (event.target.id === 'adaptivePmlEnabled') updateEnabled();
      if (typeof window.changed === 'function') window.changed();
    });
    updateEnabled();
  }

  function write(config) {
    init();
    if (!root) return;
    original = config && typeof config === 'object' && !Array.isArray(config) ? { ...config } : {};
    const value = { ...defaults, ...original };
    byId('adaptivePmlEnabled').checked = Boolean(value.enabled);
    byId('adaptivePmlTargetTail').value = percent(value.target_tail);
    byId('adaptivePmlVerify').checked = Boolean(value.verify);
    byId('adaptivePmlVerificationQuantity').value = value.verification_quantity;
    byId('adaptivePmlVerificationRtol').value = percent(value.verification_rtol);
    byId('adaptivePmlMaxSolves').value = value.max_solves;
    byId('adaptivePmlMaxM').value = value.max_M;
    byId('adaptivePmlMaxWidthFactor').value = value.max_width_factor;
    byId('adaptivePmlMaxSeconds').value = value.max_seconds;
    updateEnabled();
  }

  function read() {
    init();
    if (!root) return { ...defaults, ...original };
    return {
      ...original,
      enabled: byId('adaptivePmlEnabled').checked,
      target_tail: asNumber('adaptivePmlTargetTail') / 100,
      verify: byId('adaptivePmlVerify').checked,
      verification_quantity: byId('adaptivePmlVerificationQuantity').value,
      verification_rtol: asNumber('adaptivePmlVerificationRtol') / 100,
      max_solves: asNumber('adaptivePmlMaxSolves'),
      max_M: asNumber('adaptivePmlMaxM'),
      max_width_factor: asNumber('adaptivePmlMaxWidthFactor'),
      max_seconds: asNumber('adaptivePmlMaxSeconds')
    };
  }

  function setBusy(value) {
    busy = Boolean(value);
    updateEnabled();
  }

  const statusText = {
    attenuation_met: '边界残余达标 / 未做物理验证',
    limit_reached: '达到计算上限 / 未完成验证',
    time_limit: '达到时间预算，未完成验证'
  };
  const stageText = {
    absorption:'加厚选参',spectral_check:'提高M验证',window_check:'加厚PML验证',initial: '初始', attenuation: '吸收调参', verification: '物理区验证',
    verify_M: '提高 M 验证', verify_width: '加厚 PML 验证', final: '最终'
  };

  function appendText(parent, tag, className, content) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = content;
    parent.appendChild(element);
    return element;
  }

  function statusLabel(metadata) {
    return metadata.status === 'verified'
      ? (metadata.verification_quantity === 'electric' ? '电场矢量稳定性验证通过' : '主复场稳定性验证通过')
      : (statusText[metadata.status] || metadata.status || '自动选参结果');
  }

  function renderResult(metadata) {
    const target = document.getElementById('adaptivePmlResult');
    if (!target) return;
    target.replaceChildren();
    if (!metadata || !metadata.enabled) { target.hidden = true; return; }
    target.hidden = false;
    target.className = `apml-result apml-status-${String(metadata.status || 'unknown').replace(/[^a-z_]/gi, '')}`;
    if (!metadata.polarizations) { renderBlock(target, metadata, 'PML 自动选参', 'h3'); return; }
    // Mixed polarization: TE and TM searches alternate until both accept one common setup.
    appendText(target, 'h3', 'apml-result-title', 'PML 自动选参 · 混合偏振（TE、TM 共用同一组参数）');
    appendText(target, 'div', 'apml-status', statusLabel(metadata));
    if (metadata.message) appendText(target, 'p', 'apml-result-message', metadata.message);
    const summary = document.createElement('div');
    summary.className = 'apml-summary';
    const rounds = Array.isArray(metadata.rounds) ? metadata.rounds : [];
    if (rounds.length) appendText(summary, 'span', '', `交替顺序：${rounds.map(r => `${r.polarization}（M=${displayNumber(r.M, 0)}，${statusText[r.status] || r.status}）`).join(' → ')}`);
    if (metadata.final) appendText(summary, 'span', '', `共同参数：M=${metadata.final.M}，PML厚度${displayNumber(metadata.final.left)} / ${displayNumber(metadata.final.right)} μm`);
    if (metadata.max_solves_used != null) appendText(summary, 'span', '', `共求解 ${displayNumber(metadata.max_solves_used, 0)} 次`);
    if (summary.childElementCount) target.appendChild(summary);
    ['TE', 'TM'].forEach(pol => {
      const meta = metadata.polarizations[pol];
      if (!meta) return;
      const block = document.createElement('div');
      block.className = 'apml-pol';
      renderBlock(block, meta, `${pol} 偏振`, 'h4');
      target.appendChild(block);
    });
  }

  function renderBlock(target, metadata, title, headingTag) {
    appendText(target, headingTag, 'apml-result-title', title);
    appendText(target, 'div', 'apml-status', statusLabel(metadata));
    if (metadata.message) appendText(target, 'p', 'apml-result-message', metadata.message);

    const summary = document.createElement('div');
    summary.className = 'apml-summary';
    if (metadata.target_tail != null) appendText(summary, 'span', '', `边界对比场残余目标 ${displayPercent(metadata.target_tail)}`);
    if (metadata.tail_ratio != null) appendText(summary, 'span', '', `实际边界对比场残余 ${displayPercent(metadata.tail_ratio)}`);
    if (metadata.verify != null) appendText(summary, 'span', '', metadata.verify ? '另做物理区验证' : '未做物理区验证');
    if (metadata.verify && metadata.verification_quantity) appendText(summary, 'span', '', metadata.verification_quantity === 'electric' ? '验证场量：电场矢量' : '验证场量：主复场（TE: Ey / TM: Hy）');
    if (metadata.max_solves_used != null) appendText(summary, 'span', '', `已求解 ${displayNumber(metadata.max_solves_used, 0)} 次`);
    if(metadata.final)appendText(summary,'span','',`最终采用第${metadata.selected_solve_index}次：M=${metadata.final.M}，PML厚度${displayNumber(metadata.final.left)} / ${displayNumber(metadata.final.right)} μm`);if (summary.childElementCount) target.appendChild(summary);
    appendText(target, 'p', 'apml-result-note', '边界对比场残余是吸收诊断，不是反射率或真实误差上界。');

    const history = Array.isArray(metadata.history) ? metadata.history : [];
    if (!history.length) return;
    const details = document.createElement('details');
    details.className = 'apml-history';
    appendText(details, 'summary', '', `迭代记录（${history.length} 次）`);
    const scroll = document.createElement('div');
    scroll.className = 'apml-history-scroll';
    const table = document.createElement('table');
    const header = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ['次数', '阶段', 'M', '周期 μm', '左 / 右厚度 μm', '拉伸强度', '边界残余', '求解秒', '所选场量变化'].forEach(label => appendText(headerRow, 'th', '', label));
    header.appendChild(headerRow);
    table.appendChild(header);
    const body = document.createElement('tbody');
    history.forEach(row => {
      const tr = document.createElement('tr');if(row.solve_index===metadata.selected_solve_index)tr.className='apml-selected';
      const stage = stageText[row.stage] || row.stage || '—';
      const cells = [
        displayNumber(row.solve_index, 0), stage, displayNumber(row.M, 0),
        displayNumber(row.period, 5),
        `${displayNumber(row.left, 5)} / ${displayNumber(row.right, 5)}`,
        displayNumber(row.strength, 4),
        row.tail_ratio == null ? '—' : displayPercent(row.tail_ratio),
        displayNumber(row.solve_seconds, 2),
        row.field_change == null ? '—' : displayPercent(row.field_change)
      ];
      cells.forEach(value => appendText(tr, 'td', '', value));
      body.appendChild(tr);
    });
    table.appendChild(body);
    scroll.appendChild(table);
    details.appendChild(scroll);
    target.appendChild(details);
  }

  return { init, write, read, setBusy, renderResult };
})();
