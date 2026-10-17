"use strict";
(function () {
  const editor = {root: null, edges: [0, 1], lx: 1, ly: 1};
  const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const format = value => String(Number(value));
  const split = text => String(text ?? "").trim().split(/[,，;；\s]+/).filter(Boolean);
  const same = (a,b) => a.length === b.length && a.every((v,i) => Math.abs(v-b[i]) <= 1e-10*Math.max(1,Math.abs(v),Math.abs(b[i])));
  function dirty() {
    if (typeof changed === "function") changed();
    if (typeof layerDirty !== "undefined") layerDirty = true;
    const badge = document.getElementById("layerDirty");
    if (badge) badge.hidden = false;
  }
  function init() {
    const root = document.getElementById("layer2dEditor");
    if (!root) return;
    if (editor.root === root && root.dataset.models2dReady === "true") return;
    editor.root = root;
    root.dataset.models2dReady = "true";
    root.innerHTML = `<p class="g2-editor-note">二维材料系数使用解析积分。每个 y 区间可设置独立的 x 分段与折射率。</p><label for="g2-y-edges">二维层 y 分段边界 <span>μm，覆盖一个 y 周期</span></label><textarea id="g2-y-edges" rows="2" spellcheck="false" aria-label="二维层 y 分段边界"></textarea><button class="g2-update-y small-button" type="button">更新 y 分段</button><div class="g2-profile-rows"></div>`;
    root.addEventListener("input", event => {if (event.target.matches("textarea")) dirty();});
    root.addEventListener("click", event => {
      if (!event.target.closest(".g2-update-y")) return;
      try {updateY();} catch (error) {window.alert(error.message || String(error));}
    });
  }
  function initial(layer, lx, ly) {
    if (layer?.kind === "patterned2d" && Array.isArray(layer.y_edges) && Array.isArray(layer.x_profiles) && layer.x_profiles.length) {
      return {edges: layer.y_edges.map(Number), profiles: layer.x_profiles.map(profile => ({edges: [...profile.edges], n_values: [...profile.n_values]}))};
    }
    if (layer?.kind === "patterned" && Array.isArray(layer.edges) && Array.isArray(layer.n_values)) {
      return {edges: [0,ly], profiles: [{edges: [...layer.edges], n_values: [...layer.n_values]}]};
    }
    return {edges: [0,ly], profiles: [{edges: [0,lx], n_values: [layer?.n ?? 1.46]}]};
  }
  function render(layer, Lx, Ly) {
    init();
    if (!editor.root) return;
    editor.lx = Number(Lx);
    editor.ly = Number(Ly);
    const data = initial(layer, editor.lx, editor.ly);
    editor.edges = data.edges;
    editor.root.querySelector("#g2-y-edges").value = data.edges.map(format).join(", ");
    showRows(data.edges, data.profiles);
  }
  function showRows(edges, profiles) {
    const box = editor.root.querySelector(".g2-profile-rows");
    box.innerHTML = profiles.map((profile, i) => {
      const range = `${format(edges[i])}–${format(edges[i+1])} μm`;
      const xText = profile.xText ?? profile.edges.map(format).join(", ");
      const nText = profile.nText ?? profile.n_values.join(", ");
      return `<div class="g2-profile-row" data-profile="${i}"><h4>y ${escapeHtml(range)}</h4><div class="g2-profile-fields"><div><label for="g2-x-${i}">x 分段边界 <span>μm</span></label><textarea id="g2-x-${i}" class="g2-x-edges" rows="2" spellcheck="false" aria-label="y ${escapeHtml(range)} 的 x 分段边界">${escapeHtml(xText)}</textarea></div><div><label for="g2-n-${i}">各段折射率 n</label><textarea id="g2-n-${i}" class="g2-n-values" rows="2" spellcheck="false" aria-label="y ${escapeHtml(range)} 的折射率">${escapeHtml(nText)}</textarea></div></div></div>`;
    }).join("");
  }
  function parseEdges(text, period, axis, maxSegments) {
    const parts = split(text), values = parts.map(Number);
    if (values.length < 2 || values.length > maxSegments+1 || values.some(v => !Number.isFinite(v))) throw Error(`${axis} 边界需要 2 至 ${maxSegments+1} 个有效数字`);
    if (values.some((v,i) => i && v <= values[i-1])) throw Error(`${axis} 边界必须严格递增`);
    const tolerance = 1e-8*Math.max(1,Math.abs(period));
    if (Math.abs(values.at(-1)-values[0]-period) > tolerance) throw Error(`${axis} 首尾边界之差必须等于 ${format(period)} μm`);
    return values;
  }
  function currentProfiles(validate = false) {
    return [...editor.root.querySelectorAll(".g2-profile-row")].map((row,i) => {
      const xText = row.querySelector(".g2-x-edges").value, nText = row.querySelector(".g2-n-values").value;
      if (!validate) return {xText, nText};
      const edges = parseEdges(xText, editor.lx, `第 ${i+1} 个 y 区间的 x`, 2000);
      const n_values = split(nText);
      if (n_values.length !== edges.length-1) throw Error(`第 ${i+1} 个 y 区间的折射率数量应为 ${edges.length-1}`);
      return {edges, n_values};
    });
  }
  function remap(edges) {
    const rows = currentProfiles(false), old = editor.edges;
    const profiles = edges.slice(0,-1).map((start,i) => {
      const raw = (start+edges[i+1])/2;const mid=old[0]+((raw-old[0])%editor.ly+editor.ly)%editor.ly;
      const index = Math.max(0,Math.min(rows.length-1, old.findIndex((v,j) => j < old.length-1 && v <= mid && mid < old[j+1])));
      return rows[index];
    });
    editor.edges = edges;
    showRows(edges, profiles);
  }
  function updateY() {
    if (!editor.root) return;
    const edges = parseEdges(editor.root.querySelector("#g2-y-edges").value, editor.ly, "y", 64);
    if (!same(edges,editor.edges)) remap(edges);
    dirty();
  }
  function read(layer, Lx, Ly) {
    init();
    if (!editor.root) throw Error("二维层编辑器未就绪");
    editor.lx = Number(Lx);
    editor.ly = Number(Ly);
    const y_edges = parseEdges(editor.root.querySelector("#g2-y-edges").value, editor.ly, "y", 64);
    if (!same(y_edges,editor.edges)) remap(y_edges);
    const x_profiles = currentProfiles(true);
    if (x_profiles.length !== y_edges.length-1) throw Error("y 区间与 x 剖面数量不一致");
    const result = {...layer, kind:"patterned2d", y_edges, x_profiles};
    delete result.n;
    delete result.edges;
    delete result.n_values;
    result.fourier_method = "analytic";
    result.fft_samples = 0;
    return result;
  }
  window.Models2DEditor = {init, render, read};
})();
