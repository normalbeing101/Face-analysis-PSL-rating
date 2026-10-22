/* PSL Studio frontend — talks to the FastAPI backend on the same origin. */

const $ = (id) => document.getElementById(id);

let currentImageB64 = null;
let lastReport = null;
let activeTab = "ratios";

// ---------------------------------------------------------------- presets
async function loadPresets() {
  try {
    const r = await fetch("/api/presets");
    const data = await r.json();
    const sel = $("preset-select");
    sel.innerHTML = "";
    for (const p of data.presets) {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.label || p.name;
      opt.title = p.description || "";
      sel.appendChild(opt);
    }
    if ([...sel.options].some(o => o.value === "harmony")) sel.value = "harmony";
  } catch (e) { console.error("presets", e); }
}

// ---------------------------------------------------------------- health
async function checkHealth() {
  const pill = $("health-pill");
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    if (h.vlm_reachable && h.vlm_model) {
      pill.textContent = `VLM: ${h.vlm_model}`;
      pill.className = "pill pill-on";
    } else if (h.vlm_reachable) {
      pill.textContent = "VLM: server up, no model picked";
      pill.className = "pill pill-off";
    } else {
      pill.textContent = `VLM offline (${h.vlm_provider})`;
      pill.className = "pill pill-off";
    }
  } catch {
    pill.textContent = "backend unreachable";
    pill.className = "pill pill-off";
  }
}

// ---------------------------------------------------------------- upload
function wireUpload() {
  const zone = $("drop-zone"), inner = $("drop-inner"), input = $("file-input");
  inner.addEventListener("click", () => input.click());
  input.addEventListener("change", () => input.files[0] && readImage(input.files[0]));
  ["dragover", "dragenter"].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); inner.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); inner.classList.remove("dragover"); }));
  zone.addEventListener("drop", e => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) readImage(f);
  });
}

function readImage(file) {
  if (!file.type.startsWith("image/")) return;
  const reader = new FileReader();
  reader.onload = () => {
    currentImageB64 = reader.result;         // data-url
    const img = $("preview");
    img.src = currentImageB64;
    img.hidden = false;
    $("drop-inner").style.display = "none";
    $("btn-analyze").disabled = false;
  };
  reader.readAsDataURL(file);
}

// ---------------------------------------------------------------- analyze
async function analyzeIt() {
  if (!currentImageB64) return;
  const btn = $("btn-analyze");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>Analyzing…`;
  try {
    const body = {
      image_b64: currentImageB64,
      preset: $("preset-select").value,
      band_tolerance_scale: parseFloat($("band-scale").value),
      include_symmetry: true,
      include_golden_checks: true,
      use_vlm: $("use-vlm").checked,
      question: $("question").value.trim() || null,
    };
    const r = await fetch("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok || !data.ok) {
      showError(data.error || data.detail || "Analysis failed.");
      return;
    }
    lastReport = data.report;
    renderVerdict(data.report);
    renderAdvice(data.advice, data.advice_error);
    renderMetrics();
    if (data.overlay_png_b64) {
      const ov = $("overlay-img");
      ov.src = "data:image/png;base64," + data.overlay_png_b64;
      ov.hidden = false;
    }
    $("empty-state").hidden = true;
  } catch (e) {
    showError("Network error: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze";
  }
}

function showError(msg) {
  $("empty-state").hidden = false;
  $("empty-state").innerHTML = `<p style="color:var(--bad)">${escapeHtml(msg)}</p>
    <small>Try a clearer, frontal photo — or adjust settings.</small>`;
}

// ---------------------------------------------------------------- renderers
function renderVerdict(report) {
  const v = report.verdict;
  const card = $("verdict-card");
  card.hidden = false;
  const letter = $("tier-letter");
  const tierVal = (v.tier && (typeof v.tier === "object" ? v.tier.value : v.tier)) || "?";
  letter.textContent = tierVal;
  letter.className = "tier-letter tier-" + tierVal;
  $("rating-num").textContent = (v.numeric_rating ?? 0).toFixed(2);
  $("sub-tier").textContent = v.sub_tier || "";
  setBar("bar-angles", v.angles_score);
  setBar("bar-ratios", v.ratios_score);
  setBar("bar-symmetry", v.symmetry_score);
}

function setBar(id, score) {
  requestAnimationFrame(() => {
    $(id).style.width = Math.round(Math.max(0, Math.min(1, score || 0)) * 100) + "%";
  });
}

function renderAdvice(advice, err) {
  const card = $("advice-card");
  if (!advice) {
    card.hidden = false;
    $("advice-model").textContent = "";
    $("advice-summary").textContent = err
      ? "AI advice unavailable: " + err
      : "AI advice disabled for this run.";
    ["advice-strengths", "advice-weaknesses"].forEach(id => $(id).innerHTML = "");
    $("advice-harmony").textContent = "";
    $("advice-user-answer").hidden = true;
    return;
  }
  card.hidden = false;
  $("advice-model").textContent = `${advice.model.provider} · ${advice.model.name} · conf ${(advice.confidence * 100).toFixed(0)}%`;
  $("advice-summary").textContent = advice.summary || "";
  fillList("advice-strengths", advice.strengths);
  fillList("advice-weaknesses", advice.weaknesses);
  $("advice-harmony").textContent = advice.harmony_notes || "";
  const ua = $("advice-user-answer");
  if (advice.user_answer) { ua.textContent = advice.user_answer; ua.hidden = false; }
  else ua.hidden = true;
}

function fillList(id, items) {
  const ul = $(id);
  ul.innerHTML = "";
  (items || []).forEach(t => {
    const li = document.createElement("li");
    li.textContent = t;
    ul.appendChild(li);
  });
}

function renderMetrics() {
  if (!lastReport) return;
  $("metrics-card").hidden = false;
  const tbody = $("metrics-table").querySelector("tbody");
  tbody.innerHTML = "";

  const head = document.createElement("tr");
  const cols = activeTab === "symmetry"
    ? ["Metric", "Value"] : ["Metric", "Measured", "Ideal band", "Score", "Class"];
  cols.forEach(c => { const th = document.createElement("th"); th.textContent = c; head.appendChild(th); });
  tbody.appendChild(head);

  if (activeTab === "symmetry") {
    const s = lastReport.symmetry || {};
    Object.entries(s).forEach(([k, val]) => {
      if (typeof val === "object" || typeof val === "number" || typeof val === "string") {
        addRow(tbody, [prettyKey(k), fmt(val)]);
      }
    });
    return;
  }

  const section = lastReport[activeTab] || {};
  for (const [key, m] of Object.entries(section)) {
    if (typeof m !== "object" || m === null) continue;
    const value = m.value ?? m.degrees ?? NaN;
    const band = m.ideal || m.band || [];
    const tr = document.createElement("tr");
    tr.appendChild(cell(prettyKey(key)));
    tr.appendChild(numCell(value, activeTab === "angles" ? "°" : ""));
    tr.appendChild(cell(band.length === 2 ? `${fmt(band[0])} – ${fmt(band[1])}` : "—"));
    tr.appendChild(numCell(m.score, ""));
    tr.appendChild(classCell(m.classification));
    tbody.appendChild(tr);
  }
}

function addRow(tbody, texts) {
  const tr = document.createElement("tr");
  texts.forEach(t => tr.appendChild(cell(t)));
  tbody.appendChild(tr);
}
function cell(t) { const td = document.createElement("td"); td.textContent = t; return td; }
function numCell(v, suffix) {
  const td = document.createElement("td");
  td.className = "num";
  td.textContent = (typeof v === "number" && isFinite(v)) ? fmt(v) + suffix : "—";
  return td;
}
function classTagCls(c) {
  if (!c) return "tag";
  const l = c.toLowerCase();
  if (l.includes("ideal") || l.includes("harmon") || l.includes("strong")) return "tag tag-good";
  if (l.includes("mild") || l.includes("moderate") || l.includes("deviat")) return "tag tag-warn";
  if (l.includes("severe") || l.includes("poor") || l.includes("asym")) return "tag tag-bad";
  return "tag";
}
function classCell(c) {
  const td = document.createElement("td");
  const span = document.createElement("span");
  span.className = classTagCls(c);
  span.textContent = c || "—";
  td.appendChild(span);
  return td;
}

const fmt = (x) => typeof x === "number" && isFinite(x)
  ? (Math.abs(x) >= 100 ? x.toFixed(1) : x.toFixed(3).replace(/0+$/, "").replace(/\.$/, ""))
  : String(x);
const prettyKey = (k) => k.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
const escapeHtml = (s) => s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------------------------------------------------------- tabs
function wireTabs() {
  document.querySelectorAll(".seg button").forEach(b =>
    b.addEventListener("click", () => {
      document.querySelectorAll(".seg button").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      activeTab = b.dataset.tab;
      renderMetrics();
    }));
}

// ---------------------------------------------------------------- settings
async function openSettings() {
  $("settings-modal").showModal();
  $("test-result").textContent = "";
  try {
    const cfg = await (await fetch("/api/vlm/config")).json();
    $("cfg-provider").value = cfg.provider;
    $("cfg-url").value = cfg.base_url;
    $("cfg-key").value = cfg.api_key || "";
    $("cfg-model").innerHTML = `<option value="${escapeHtml(cfg.model_name || "")}">${escapeHtml(cfg.model_name || "— load models first —")}</option>`;
    const v = cfg.vram || {};
    $("vram-mode").value = v.mode || "auto";
    $("vram-budget").value = v.vram_budget_gb ?? 8;
    $("vram-layers").value = v.n_gpu_layers ?? -1;
    $("ctx-len").value = v.context_length ?? 4096;
    $("cpu-threads").value = v.cpu_offload_threads ?? 4;
    $("kv4").checked = !!v.kv_cache_4bit;
    $("flash").checked = v.flash_attention !== false;
    refreshModels();
  } catch (e) { $("test-result").textContent = "Could not load config: " + e.message; }
}

async function refreshModels() {
  const note = $("model-note");
  note.textContent = "Querying provider…";
  try {
    const r = await fetch("/api/vlm/models");
    const data = await r.json();
    const sel = $("cfg-model");
    const prev = sel.value;
    sel.innerHTML = "";
    if (!data.reachable) {
      note.textContent = `Provider not reachable at ${data.base_url}. Start it (e.g. \`ollama serve\`), then refresh.`;
      return;
    }
    let visionCount = 0;
    for (const m of data.models) {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.name + (m.size_gb ? ` (${m.size_gb} GB)` : "") + (m.vision ? "" : " — text-only ⚠");
      if (!m.vision) opt.style.color = "#b0b0b8";
      else visionCount++;
      sel.appendChild(opt);
    }
    if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
    note.textContent = visionCount
      ? `${visionCount} vision-capable model(s) found. Text-only models cannot analyze photos.`
      : "No vision-capable models found. Pull one, e.g. `ollama pull llama3.2-vision` or `ollama pull minicpm-v`.";
  } catch (e) { note.textContent = "Model list failed: " + e.message; }
}

async function saveSettings() {
  const payload = {
    provider: $("cfg-provider").value,
    base_url: $("cfg-url").value.trim(),
    api_key: $("cfg-key").value.trim(),
    model_name: $("cfg-model").value,
    temperature: 0.2, max_tokens: 900, timeout_s: 120,
    num_parallel: 1, keep_alive: "5m",
    vram: {
      mode: $("vram-mode").value,
      vram_budget_gb: parseFloat($("vram-budget").value) || 8,
      n_gpu_layers: parseInt($("vram-layers").value, 10),
      context_length: parseInt($("ctx-len").value, 10) || 4096,
      kv_cache_4bit: $("kv4").checked,
      flash_attention: $("flash").checked,
      cpu_offload_threads: parseInt($("cpu-threads").value, 10) || 4,
    },
  };
  if (!Number.isFinite(payload.vram.n_gpu_layers)) payload.vram.n_gpu_layers = -1;
  const r = await fetch("/api/vlm/config", {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    $("test-result").textContent = "Save failed: " + (e.detail || r.status);
    return;
  }
  $("settings-modal").close();
  checkHealth();
}

async function testVlm() {
  $("test-result").textContent = "Testing… (first call loads the model into VRAM, can take a while)";
  try {
    const r = await fetch("/api/vlm/test", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "Reply with the single word: OK" }),
    });
    const d = await r.json();
    $("test-result").textContent = r.ok && d.ok
      ? `✓ ${d.model}: “${String(d.reply).slice(0, 80)}”`
      : "✗ " + (d.detail || "failed");
  } catch (e) { $("test-result").textContent = "✗ " + e.message; }
}

// ---------------------------------------------------------------- boot
window.addEventListener("DOMContentLoaded", () => {
  loadPresets();
  checkHealth();
  wireUpload();
  wireTabs();
  $("btn-analyze").addEventListener("click", analyzeIt);
  $("band-scale").addEventListener("input", e => $("band-out").textContent = (+e.target.value).toFixed(2) + "×");
  $("btn-settings").addEventListener("click", openSettings);
  $("btn-close-settings").addEventListener("click", () => $("settings-modal").close());
  $("btn-save-settings").addEventListener("click", saveSettings);
  $("btn-refresh-models").addEventListener("click", refreshModels);
  $("btn-test-vlm").addEventListener("click", testVlm);
});
