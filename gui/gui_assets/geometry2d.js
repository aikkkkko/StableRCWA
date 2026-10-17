"use strict";
(function () {
  const state = {root: null, geometry: null, selected: 0, part: "real", plane: "xz", at: 0, observer: null};
  const safe = value => String(value ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = value => Number.isFinite(Number(value)) ? Number(value) : 0;
  const label = (value, digits = 3) => Number(value).toLocaleString("en-US", {maximumFractionDigits: digits, useGrouping: false});
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const span = (g, axis) => Math.max(0, number(g?.[axis + "1"]) - number(g?.[axis + "0"]));
  function color(value, lo, hi) {
    const t = clamp(hi === lo ? 0 : (number(value) - lo) / (hi - lo), 0, 1);
    const a = [237,244,249], b = [46,105,168];
    return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
  }
  function init() {
    const root = document.getElementById("geometry2dPanel");
    if (!root) return;
    if (state.root === root && root.dataset.geometry2dReady === "true") return;
    if (state.observer) state.observer.disconnect();
    state.root = root;
    root.dataset.geometry2dReady = "true";
    root.innerHTML = `<div class="g2-head"><div><h3>二维结构视图</h3><p class="g2-subtitle">层内俯视与贯穿全部层的联动剖面</p></div><div class="g2-controls"><label>剖面方向<select class="g2-plane" aria-label="剖面方向"><option value="xz">xz · 固定 y</option><option value="yz">yz · 固定 x</option></select></label><label class="g2-at-label">y / μm<input class="g2-at" type="number" min="0" step="any" inputmode="decimal" aria-label="剖切坐标"></label></div></div><div class="g2-slider-line"><span class="g2-start">0</span><input class="g2-slider" type="range" min="0" max="1000" step="1" aria-label="移动剖切位置"><span class="g2-end">0</span></div><div class="g2-views"><section class="g2-view"><div class="g2-view-title">xy 俯视 · <span class="g2-layer-name"></span></div><div class="g2-top-svg"></div></section><section class="g2-view"><div class="g2-view-title"><span class="g2-section-title">xz 剖面</span> · 全部层</div><div class="g2-section-svg"></div></section></div><div class="g2-footer"><span class="g2-legend-title"></span><span class="g2-legend-bar" aria-hidden="true"></span><span class="g2-legend-range"></span><span class="g2-note">橙色表示当前层及剖切位置</span></div>`;
    const plane = root.querySelector(".g2-plane"), at = root.querySelector(".g2-at"), slider = root.querySelector(".g2-slider");
    plane.addEventListener("change", () => {state.plane = plane.value === "yz" ? "yz" : "xz"; state.at = clamp(state.at, 0, span(state.geometry, state.plane === "xz" ? "y" : "x")); syncControls(); draw();root.dispatchEvent(new CustomEvent("slicechange"));});
    slider.addEventListener("input", () => {const length = span(state.geometry, state.plane === "xz" ? "y" : "x"); state.at = length * number(slider.value) / 1000; syncControls(); draw();root.dispatchEvent(new CustomEvent("slicechange"));});
    at.addEventListener("input", () => {if (at.value === "") return; const value = Number(at.value); if (!Number.isFinite(value)) return; state.at = clamp(value, 0, span(state.geometry, state.plane === "xz" ? "y" : "x")); syncControls(false); draw();root.dispatchEvent(new CustomEvent("slicechange"));});
    at.addEventListener("change", () => syncControls());
    if (typeof ResizeObserver !== "undefined") {
      state.observer = new ResizeObserver(() => {if (state.geometry) draw();});
      state.observer.observe(root.querySelector(".g2-views"));
    }
    syncControls();
    if (state.geometry) draw();
  }
  function syncControls(updateNumber = true) {
    const root = state.root;
    if (!root) return;
    const axis = state.plane === "xz" ? "y" : "x", length = span(state.geometry, axis);
    root.querySelector(".g2-plane").value = state.plane;
    root.querySelector(".g2-at-label").firstChild.textContent = `${axis} / μm`;
    const at = root.querySelector(".g2-at");
    at.max = String(length);
    if (updateNumber) at.value = label(state.at, 6);
    root.querySelector(".g2-slider").value = length ? String(Math.round(1000 * state.at / length)) : "0";
    root.querySelector(".g2-start").textContent = "0";
    root.querySelector(".g2-end").textContent = label(length, 4);
  }
  function render(geometry, selectedLayer, part = "real") {
    init();
    if (!state.root || !geometry) return;
    if (state.geometry !== geometry) {
      const sampling = geometry.config?.sampling || {};
      state.plane = sampling.slice_plane === "yz" ? "yz" : "xz";
      state.at = Number.isFinite(Number(sampling.slice_at)) ? Number(sampling.slice_at) : 0;
    }
    state.geometry = geometry;
    state.selected = clamp(Math.trunc(number(selectedLayer)), 0, Math.max(0, (geometry.layers || []).length - 1));
    state.part = part === "imag" ? "imag" : "real";
    state.at = clamp(state.at, 0, span(geometry, state.plane === "xz" ? "y" : "x"));
    syncControls();
    draw();
  }
  function getSlice() {return {slice_plane: state.plane, slice_at: state.at};}
  function scales(g) {
    const values = (g.layers || []).flatMap(layer => (layer.rectangles || []).map(rect => number(rect[`epsilon_${state.part}`])));
    if (!values.length) values.push(0);
    const lo = Math.min(...values), hi = Math.max(...values);
    return {lo, hi, paint: value => color(value, lo, hi)};
  }
  function size(node, minimum) {
    return Math.max(minimum, Math.round(node.getBoundingClientRect().width || node.clientWidth || node.parentElement?.clientWidth || minimum));
  }
  function wrap(position, length) {
    if (!(length > 0)) return 0;
    const value = position % length;
    return value < 0 ? value + length : value;
  }
  function cuts(layer, plane, at, g) {
    const lateral = plane === "xz" ? "y" : "x", horizontal = plane === "xz" ? "x" : "y";
    const origin = number(g[lateral + "0"]), length = span(g, lateral);
    const position = origin + wrap(at - origin, length);
    const start = number(g[horizontal + "0"]), end = number(g[horizontal + "1"]);
    const matching = (layer.rectangles || []).filter(r => number(r[lateral + "0"]) <= position && position < number(r[lateral + "1"]));
    const edges = [...new Set([start, end, ...matching.flatMap(r => [clamp(number(r[horizontal + "0"]), start, end), clamp(number(r[horizontal + "1"]), start, end)])])].sort((a,b) => a-b);
    const pieces = [];
    for (let i = 0; i < edges.length - 1; i++) {
      const a = edges[i], b = edges[i+1];
      if (b <= a) continue;
      const mid = (a+b)/2;
      const rect = matching.findLast ? matching.findLast(r => number(r[horizontal + "0"]) <= mid && mid < number(r[horizontal + "1"])) : matching.slice().reverse().find(r => number(r[horizontal + "0"]) <= mid && mid < number(r[horizontal + "1"]));
      pieces.push({a, b, rect});
    }
    return pieces;
  }
  function draw() {
    const root = state.root, g = state.geometry;
    if (!root || !g || !Array.isArray(g.layers) || !g.layers.length) return;
    const scale = scales(g), layer = g.layers[state.selected];
    root.querySelector(".g2-layer-name").textContent = layer.name || `第 ${state.selected+1} 层`;
    root.querySelector(".g2-section-title").textContent = `${state.plane} 剖面`;
    root.querySelector(".g2-legend-title").textContent = `${state.part === "real" ? "Re" : "Im"} εᵣ`;
    root.querySelector(".g2-legend-range").textContent = `${label(scale.lo)} – ${label(scale.hi)}`;
    drawTop(root.querySelector(".g2-top-svg"), g, layer, scale);
    drawSection(root.querySelector(".g2-section-svg"), g, scale);
  }
  function drawTop(node, g, layer, scale) {
    const W = size(node, 220), H = Math.max(270, Math.min(490, Math.round(W * .59)));
    const m = {l:48,r:18,t:19,b:39}, pw = W-m.l-m.r, ph = H-m.t-m.b;
    const x0 = number(g.x0), y0 = number(g.y0), lx = span(g,"x") || 1, ly = span(g,"y") || 1;
    const xx = x => m.l + (number(x)-x0)/lx*pw, yy = y => m.t + (number(y)-y0)/ly*ph;
    const axis = state.plane === "xz" ? "y" : "x", cut = number(g[axis+"0"]) + state.at;
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${safe(layer.name)}的xy材料俯视图"><rect x="${m.l}" y="${m.t}" width="${pw}" height="${ph}" fill="#f8fafc"/>`;
    for (const r of layer.rectangles || []) {
      const x = xx(r.x0), y = yy(r.y0), w = xx(r.x1)-x, h = yy(r.y1)-y;
      if (w <= 0 || h <= 0) continue;
      svg += `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${scale.paint(r[`epsilon_${state.part}`])}" stroke="#8fa8bf" stroke-width=".6"><title>${safe(layer.name)} · n=${safe(r.n)} · ε=${label(r.epsilon_real)}${number(r.epsilon_imag) ? ` + ${label(r.epsilon_imag)}i` : ""}</title></rect>`;
    }
    svg += state.plane === "xz" ? `<line x1="${m.l}" x2="${m.l+pw}" y1="${yy(cut)}" y2="${yy(cut)}" stroke="#dc8a26" stroke-width="2.5" stroke-dasharray="7 4" pointer-events="none"/>` : `<line x1="${xx(cut)}" x2="${xx(cut)}" y1="${m.t}" y2="${m.t+ph}" stroke="#dc8a26" stroke-width="2.5" stroke-dasharray="7 4" pointer-events="none"/>`;
    svg += `<rect x="${m.l}" y="${m.t}" width="${pw}" height="${ph}" fill="none" stroke="#9fb2c7"/>`;
    for (let i=0;i<=4;i++) svg += `<text x="${m.l+pw*i/4}" y="${H-15}" text-anchor="middle" font-size="12" fill="#667b91">${label(x0+lx*i/4)}</text><text x="${m.l-7}" y="${m.t+ph*i/4+4}" text-anchor="end" font-size="12" fill="#667b91">${label(y0+ly*i/4)}</text>`;
    svg += `<text x="${m.l+pw/2}" y="${H-1}" text-anchor="middle" font-size="12" fill="#4d647f">x / μm</text><text x="13" y="${m.t+ph/2}" text-anchor="middle" font-size="12" fill="#4d647f" transform="rotate(-90 13 ${m.t+ph/2})">y / μm</text></svg>`;
    node.innerHTML = svg;
  }
  function drawSection(node, g, scale) {
    const W = size(node, 220), H = Math.max(270, Math.min(490, Math.round(W * .59)));
    const m = {l:48,r:18,t:19,b:39}, pw = W-m.l-m.r, ph = H-m.t-m.b;
    const axis = state.plane === "xz" ? "x" : "y", a0 = number(g[axis+"0"]), length = span(g,axis) || 1;
    const z0 = Math.min(0, ...g.layers.map(l => number(l.z0))), z1 = Math.max(number(g.depth), ...g.layers.map(l => number(l.z1)));
    const depth = Math.max(z1-z0, 1e-9), xx = x => m.l+(number(x)-a0)/length*pw, zz = z => m.t+(number(z)-z0)/depth*ph;
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${state.plane}材料剖面，${safe(g.layers[state.selected].name)}为当前层"><rect x="${m.l}" y="${m.t}" width="${pw}" height="${ph}" fill="#f8fafc"/>`;
    g.layers.forEach((layer, index) => {
      const top = zz(layer.z0), bottom = zz(layer.z1);
      if (bottom <= top) return;
      for (const piece of cuts(layer, state.plane, state.at, g)) {
        const left = xx(piece.a), width = xx(piece.b)-left;
        const fill = piece.rect ? scale.paint(piece.rect[`epsilon_${state.part}`]) : "#f8fafc";
        svg += `<rect x="${left}" y="${top}" width="${width}" height="${bottom-top}" fill="${fill}" stroke="#9fb2c7" stroke-width=".5">${piece.rect ? `<title>${safe(layer.name)} · n=${safe(piece.rect.n)} · ε=${label(piece.rect.epsilon_real)}${number(piece.rect.epsilon_imag) ? ` + ${label(piece.rect.epsilon_imag)}i` : ""}</title>` : ""}</rect>`;
      }
      if (index === state.selected) svg += `<rect x="${m.l}" y="${top}" width="${pw}" height="${bottom-top}" fill="none" stroke="#dc8a26" stroke-width="2" pointer-events="none"/><text x="${m.l+6}" y="${clamp(top+15,m.t+13,m.t+ph-4)}" fill="#a65f0c" font-size="11" font-weight="600" pointer-events="none">${safe(layer.name || `第 ${index+1} 层`)}</text>`;
    });
    svg += `<rect x="${m.l}" y="${m.t}" width="${pw}" height="${ph}" fill="none" stroke="#9fb2c7"/>`;
    for (let i=0;i<=4;i++) svg += `<text x="${m.l+pw*i/4}" y="${H-15}" text-anchor="middle" font-size="12" fill="#667b91">${label(a0+length*i/4)}</text><text x="${m.l-7}" y="${m.t+ph*i/4+4}" text-anchor="end" font-size="12" fill="#667b91">${label(z0+depth*i/4)}</text>`;
    svg += `<text x="${m.l+pw/2}" y="${H-1}" text-anchor="middle" font-size="12" fill="#4d647f">${axis} / μm</text><text x="13" y="${m.t+ph/2}" text-anchor="middle" font-size="12" fill="#4d647f" transform="rotate(-90 13 ${m.t+ph/2})">z / μm</text></svg>`;
    node.innerHTML = svg;
  }
  window.Geometry2D = {init, render, getSlice};
})();
