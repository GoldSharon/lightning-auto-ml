/* ─── Lightning AutoML — Frontend SPA ───────────────────────────────────────── */

const API = '';   // same-origin

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
  stage:          'empty',   // empty | uploaded | configured | analyzed | trained
  datasetInfo:    null,
  config:         {},
  insights:       null,
  preprocessPlan: null,
  trainResults:   null,
  featureNames:   [],
  predHistory:    [],
};

// ─── DOM helpers ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const show = id => { const el = $(id); if (el) el.classList.remove('hidden'); };
const hide = id => { const el = $(id); if (el) el.classList.add('hidden'); };

function toast(msg, type = 'info') {
  const c = $('toast-container');
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

function setScreen(name) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const target = document.querySelector(`[data-screen="${name}"]`);
  if (target) target.classList.add('active');

  // Update step nav
  const steps = ['home', 'upload', 'analyze', 'insights', 'preprocess', 'training', 'results', 'predict', 'download'];
  const idx = steps.indexOf(name);
  document.querySelectorAll('.step-item').forEach((el, i) => {
    el.classList.remove('active', 'done');
    if (i === idx) el.classList.add('active');
    else if (i < idx) el.classList.add('done');
  });
}

function updateSessionBadge() {
  const badge = $('session-badge');
  if (state.stage !== 'empty') {
    badge.textContent = state.datasetInfo?.filename || 'Session active';
    badge.classList.add('visible');
  } else {
    badge.classList.remove('visible');
  }
}

// ─── API helpers ──────────────────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const res = await fetch(API + path, options);
  const data = await res.json();
  if (!res.ok) {
    const msg = data.detail || data.message || 'Request failed';
    throw new Error(msg);
  }
  return data;
}

// ─── Screen: HOME ─────────────────────────────────────────────────────────────
function initHome() {
  $('btn-start').addEventListener('click', () => setScreen('upload'));

  // Restore session if exists
  apiFetch('/api/session').then(data => {
    if (data.has_session && data.stage !== 'empty') {
      state.stage = data.stage;
      state.config = {
        learning_type:  data.learning_type,
        target_column:  data.target_column,
        project_name:   data.project_name,
      };
      updateSessionBadge();

      // Show restore banner
      const banner = $('restore-banner');
      if (banner) {
        banner.querySelector('.restore-stage').textContent = data.stage;
        banner.querySelector('.restore-project').textContent = data.project_name || '—';
        show('restore-banner');
      }
    }
  }).catch(() => {});
}

$('btn-restore')?.addEventListener('click', () => {
  if (state.stage === 'trained') setScreen('results');
  else if (state.stage === 'analyzed') setScreen('preprocess');
  else if (state.stage === 'configured') setScreen('analyze');
  else setScreen('upload');
});

// ─── Screen: UPLOAD ───────────────────────────────────────────────────────────
function initUpload() {
  const zone  = $('upload-zone');
  const input = $('file-input');

  $('btn-open-upload').addEventListener('click', () => input.click());
  zone.addEventListener('click', () => input.click());

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });

  input.addEventListener('change', () => { if (input.files[0]) handleFile(input.files[0]); });
}

async function handleFile(file) {
  const allowed = ['csv', 'xlsx', 'xls', 'json'];
  const ext = file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    toast(`File type .${ext} not supported. Use: CSV, Excel, JSON`, 'error');
    return;
  }

  $('upload-zone').innerHTML = `<div class="spinner" style="margin:0 auto"></div><p class="text-muted mt-4">Uploading ${file.name}…</p>`;

  const fd = new FormData();
  fd.append('file', file);

  try {
    const data = await apiFetch('/api/upload', { method: 'POST', body: fd });
    state.datasetInfo = data;
    state.stage = 'uploaded';
    updateSessionBadge();
    renderDatasetInfo(data);
    setScreen('upload');
    toast(`${data.filename} uploaded — ${data.rows} rows`, 'success');
  } catch (err) {
    toast(err.message, 'error');
    resetUploadZone();
  }
}

function resetUploadZone() {
  $('upload-zone').innerHTML = `
    <div class="upload-icon">📂</div>
    <div class="upload-title">Drop your dataset here</div>
    <div class="upload-hint">CSV · Excel · JSON &nbsp;·&nbsp; Max 50 MB</div>`;
}

function renderDatasetInfo(info) {
  show('dataset-info-section');

  $('ds-filename').textContent  = info.filename;
  $('ds-rows').textContent      = info.rows.toLocaleString();
  $('ds-cols').textContent      = info.columns;
  $('ds-missing').textContent   = Object.values(info.missing_counts).reduce((a, b) => a + b, 0);

  // Populate target column dropdown
  const sel = $('target-col-select');
  sel.innerHTML = '<option value="">— None (Clustering) —</option>';
  info.column_names.forEach(col => {
    const opt = document.createElement('option');
    opt.value = col;
    opt.textContent = col;
    if (col === info.suggested_target) opt.selected = true;
    sel.appendChild(opt);
  });

  // Set detected task
  if (info.detected_task) {
    $('task-select').value = info.detected_task;
    updateMLType(info.detected_task);
  }

  // Render preview table
  renderPreviewTable(info.sample_data, info.column_names);

  // Scroll to config
  $('config-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderPreviewTable(rows, cols) {
  const wrap = $('preview-table-wrap');
  if (!rows.length) return;
  let html = '<table class="preview-table"><thead><tr>';
  cols.forEach(c => html += `<th>${c}</th>`);
  html += '</tr></thead><tbody>';
  rows.forEach(row => {
    html += '<tr>';
    cols.forEach(c => html += `<td title="${row[c]}">${row[c] ?? ''}</td>`);
    html += '</tr>';
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

function updateMLType(taskVal) {
  const mlType = $('ml-type-select');
  if (taskVal === 'Clustering') mlType.value = 'Unsupervised';
  else mlType.value = 'Supervised';
}

$('task-select')?.addEventListener('change', e => updateMLType(e.target.value));

$('btn-confirm-config')?.addEventListener('click', async () => {
  const target    = $('target-col-select').value;
  const learning  = $('task-select').value;
  const mlType    = $('ml-type-select').value;
  const testSize  = parseFloat($('test-size').value) || 0.2;
  const projName  = $('project-name').value.trim() || 'automl_project';

  if (mlType === 'Supervised' && !target) {
    toast('Select a target column for supervised learning.', 'error');
    return;
  }

  try {
    const data = await apiFetch('/api/configure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_column:  target || null,
        ml_type:        mlType,
        learning_type:  learning,
        test_size:      testSize,
        project_name:   projName,
      }),
    });
    state.config = data.config;
    state.stage  = 'configured';
    toast('Pipeline configured ✓', 'success');
    setScreen('analyze');
    startAnalysis();
  } catch (err) {
    toast(err.message, 'error');
  }
});

// ─── Screen: ANALYZE (progress) ───────────────────────────────────────────────
const ANALYSIS_STEPS = [
  { id: 'ps-summary',  label: 'Dataset summary'         },
  { id: 'ps-dtypes',   label: 'Data type assignment'     },
  { id: 'ps-features', label: 'Feature analysis'         },
  { id: 'ps-report',   label: 'Building analysis report' },
  { id: 'ps-plan',     label: 'Generating cleaning plan' },
];

const TRAINING_STEPS = [
  { id: 'ts-drop',     label: 'Dropping columns'         },
  { id: 'ts-impute',   label: 'Missing value imputation' },
  { id: 'ts-fe',       label: 'Feature engineering'      },
  { id: 'ts-outliers', label: 'Outlier handling'         },
  { id: 'ts-encode',   label: 'Categorical encoding'     },
  { id: 'ts-scale',    label: 'Scaling & transformation' },
  { id: 'ts-train',    label: 'Training ML models'       },
];

function renderProgressSteps(containerId, steps) {
  const c = $(containerId);
  if (!c) return;
  c.innerHTML = steps.map((s, i) => `
    <div class="progress-step" id="${s.id}">
      <div class="step-icon">${i + 1}</div>
      <span>${s.label}</span>
    </div>`).join('');
}

function setStepDone(id) {
  const el = $(id);
  if (!el) return;
  el.classList.remove('active');
  el.classList.add('done');
  el.querySelector('.step-icon').textContent = '✓';
}

function setStepActive(id) {
  const el = $(id);
  if (!el) return;
  el.classList.add('active');
  el.querySelector('.step-icon').innerHTML = '<div class="spinner" style="width:12px;height:12px;border-width:1.5px"></div>';
}

function setProgress(pct, label = '') {
  const bar = $('progress-bar');
  const pctEl = $('progress-pct');
  const labelEl = $('progress-label');
  if (bar) bar.style.width = pct + '%';
  if (pctEl) pctEl.textContent = pct + '%';
  if (labelEl) labelEl.textContent = label;
}

async function startAnalysis() {
  renderProgressSteps('analysis-steps', ANALYSIS_STEPS);
  setProgress(0, 'Starting analysis…');

  // Simulate step progression while real API runs
  const stepIds = ANALYSIS_STEPS.map(s => s.id);
  let stepIdx = 0;

  const interval = setInterval(() => {
    if (stepIdx < stepIds.length) {
      if (stepIdx > 0) setStepDone(stepIds[stepIdx - 1]);
      setStepActive(stepIds[stepIdx]);
      setProgress(Math.round(((stepIdx + 1) / (stepIds.length + 1)) * 80));
      stepIdx++;
    }
  }, 1200);

  try {
    const data = await apiFetch('/api/analyze', { method: 'POST' });
    clearInterval(interval);

    stepIds.forEach(id => setStepDone(id));
    setProgress(100, 'Analysis complete!');

    state.insights      = data.insights;
    state.stage         = 'analyzed';
    state.preprocessPlan = data.insights?.preprocessing_plan;

    setTimeout(() => {
      setScreen('insights');
      renderInsights(state.insights);
    }, 800);
    toast('Analysis complete ✓', 'success');
  } catch (err) {
    clearInterval(interval);
    setProgress(0, 'Analysis failed.');
    toast(err.message, 'error');
  }
}

// ─── Screen: INSIGHTS ─────────────────────────────────────────────────────────
function renderInsights(ins) {
  if (!ins) return;

  // Quality score
  const score = Math.round(ins.quality_score || 0);
  $('quality-score').textContent = score;
  $('quality-score-sub').textContent = score >= 80 ? 'Excellent' : score >= 60 ? 'Good' : 'Needs attention';

  // Key features
  const featsEl = $('key-features');
  featsEl.innerHTML = (ins.key_features || []).map(f =>
    `<span class="feat-tag high">${f}</span>`
  ).join('');

  // Dropped columns
  const dropEl = $('dropped-cols');
  const dropped = ins.dropped_columns || [];
  if (dropped.length) {
    dropEl.innerHTML = dropped.map(c => {
      const reasons = ins.drop_reasons?.[c] || [];
      return `<span class="feat-tag dropped" title="${reasons.join('; ')}">${c}</span>`;
    }).join('');
  } else {
    dropEl.innerHTML = '<span class="text-muted">No columns dropped</span>';
  }

  // Missing values bar chart
  const missing = ins.missing_values || {};
  const missingEntries = Object.entries(missing).filter(([, v]) => v > 0).slice(0, 10);
  renderBarChart('missing-chart', missingEntries, '#ef4444', v => `${v.toFixed(1)}%`);

  // Outliers bar chart
  const outliers = ins.outliers_detected || {};
  const outlierEntries = Object.entries(outliers).filter(([, v]) => v > 0).slice(0, 10);
  renderBarChart('outliers-chart', outlierEntries, '#f59e0b', v => v);

  // Correlations
  const corrs = ins.correlations || {};
  const corrEntries = Object.entries(corrs).slice(0, 8);
  renderBarChart('corr-chart', corrEntries, '#6366f1', v => v.toFixed(2), true);
}

function renderBarChart(containerId, entries, color, fmtVal, centerZero = false) {
  const c = $(containerId);
  if (!c) return;

  if (!entries.length) {
    c.innerHTML = '<span class="text-muted">No data</span>';
    return;
  }

  const maxVal = Math.max(...entries.map(([, v]) => Math.abs(v)));
  c.innerHTML = `<div class="bar-chart">${entries.map(([label, val]) => {
    const pct = maxVal > 0 ? Math.round((Math.abs(val) / maxVal) * 100) : 0;
    return `
      <div class="bar-row">
        <span class="bar-label" title="${label}">${label.length > 14 ? label.slice(0,12)+'…' : label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="bar-val">${fmtVal(val)}</span>
      </div>`;
  }).join('')}</div>`;
}

$('btn-to-preprocess')?.addEventListener('click', async () => {
  setScreen('preprocess');
  await loadPreprocessPlan();
});

// ─── Screen: PREPROCESSING ────────────────────────────────────────────────────
async function loadPreprocessPlan() {
  try {
    const data = await apiFetch('/api/preprocessing');
    const plan = data.plan;
    renderPreprocessPlan(plan);
  } catch (err) {
    toast(err.message, 'error');
  }
}

function renderPreprocessPlan(plan) {
  $('plan-missing').textContent  = plan.missing_strategy  || '—';
  $('plan-outliers').textContent = plan.outlier_method    || '—';
  $('plan-scaling').textContent  = plan.scaling_method    || '—';

  const encParts = [];
  if (plan.onehot_columns?.length)     encParts.push(`One-hot (${plan.onehot_columns.length} cols)`);
  if (plan.frequency_columns?.length)  encParts.push(`Frequency (${plan.frequency_columns.length} cols)`);
  $('plan-encoding').textContent = encParts.join(', ') || 'None';

  // Feature engineering
  const feEl = $('plan-fe');
  const feList = plan.feature_engineering || [];
  if (feList.length) {
    feEl.innerHTML = feList.map(f =>
      `<div class="plan-item"><div class="plan-item-label">${f.new_feature}</div><div class="plan-item-value" style="font-size:0.75rem;color:var(--text-dim)">${f.description || '—'}</div></div>`
    ).join('');
  } else {
    feEl.innerHTML = '<span class="text-muted">No feature engineering suggestions</span>';
  }

  // Insights
  const insEl = $('plan-insights');
  const insights = plan.key_insights || [];
  insEl.innerHTML = insights.map(i => `<li>${i}</li>`).join('') || '<li class="text-muted">—</li>';

  $('plan-orig-feat').textContent  = plan.original_features  || '—';
  $('plan-rec-feat').textContent   = plan.recommended_features || '—';
}

$('btn-accept-plan')?.addEventListener('click', () => {
  setScreen('training');
  startTraining();
});

// ─── Screen: TRAINING ─────────────────────────────────────────────────────────
async function startTraining() {
  renderProgressSteps('training-steps', TRAINING_STEPS);
  setProgress(0, 'Starting preprocessing & training…', 'train-progress-bar', 'train-progress-pct', 'train-progress-label');

  const stepIds = TRAINING_STEPS.map(s => s.id);
  let stepIdx = 0;
  const interval = setInterval(() => {
    if (stepIdx < stepIds.length) {
      if (stepIdx > 0) setStepDone(stepIds[stepIdx - 1]);
      setStepActive(stepIds[stepIdx]);
      const pct = Math.round(((stepIdx + 1) / (stepIds.length + 1)) * 85);
      const trainBar = $('train-progress-bar');
      const trainPct = $('train-progress-pct');
      if (trainBar) trainBar.style.width = pct + '%';
      if (trainPct) trainPct.textContent = pct + '%';
      stepIdx++;
    }
  }, 2500);

  try {
    const data = await apiFetch('/api/train', { method: 'POST' });
    clearInterval(interval);
    stepIds.forEach(id => setStepDone(id));

    const trainBar = $('train-progress-bar');
    const trainPct = $('train-progress-pct');
    const trainLbl = $('train-progress-label');
    if (trainBar) trainBar.style.width = '100%';
    if (trainPct) trainPct.textContent = '100%';
    if (trainLbl) trainLbl.textContent = 'Training complete!';

    state.trainResults = data.results;
    state.stage = 'trained';

    setTimeout(() => {
      setScreen('results');
      renderResults(data.results);
    }, 800);
    toast('Training complete ✓', 'success');
  } catch (err) {
    clearInterval(interval);
    toast(err.message, 'error');
  }
}

// ─── Screen: RESULTS ──────────────────────────────────────────────────────────
function renderResults(results) {
  if (!results) return;

  $('best-model-name').textContent  = results.best_model_name || '—';
  $('best-model-score').textContent = `Best Score: ${results.best_score?.toFixed(4) ?? '—'}  |  Training time: ${results.training_time?.toFixed(1) ?? '—'}s`;

  // Metrics grid
  const best = results.all_results?.[0];
  if (best) {
    const metricsEl = $('metrics-grid');
    metricsEl.innerHTML = Object.entries(best.metrics)
      .filter(([k]) => k !== 'fit_time')
      .slice(0, 8)
      .map(([k, v]) => `
        <div class="metric-box">
          <span class="val">${typeof v === 'number' ? v.toFixed(4) : v}</span>
          <span class="key">${k.replace(/_/g, ' ')}</span>
        </div>`).join('');
  }

  // LLM analysis
  const llm = results.llm_analysis || {};
  if (llm.analysis || llm.recommendations) {
    show('llm-analysis-section');
    $('llm-best-model').textContent    = llm.best_model    || results.best_model_name || '—';
    $('llm-analysis').textContent      = llm.analysis      || '—';
    $('llm-recommendations').textContent = llm.recommendations || '—';
  }

  // Model comparison table
  const tbody = $('models-tbody');
  tbody.innerHTML = (results.all_results || []).map((r, i) => {
    const primary = Object.entries(r.metrics).find(([k]) => k.startsWith('val_') || k === 'silhouette_score');
    const score = primary ? primary[1] : '—';
    const isBest = i === 0;
    return `<tr class="${isBest ? 'best' : ''}">
      <td>${r.model_name}${isBest ? '<span class="best-badge">best</span>' : ''}</td>
      <td>${typeof score === 'number' ? score.toFixed(4) : score}</td>
      <td>${r.metrics.fit_time?.toFixed(3) ?? '—'}s</td>
    </tr>`;
  }).join('');

  // Chart
  renderModelChart(results.all_results || []);
}

function renderModelChart(models) {
  const canvas = $('model-chart');
  if (!canvas || !models.length) return;

  const labels = models.map(m => m.model_name.replace('Regressor', '').replace('Classifier', ''));
  const primary = Object.keys(models[0].metrics).find(k => k.startsWith('val_') || k === 'silhouette_score') || 'val_r2';
  const vals = models.map(m => m.metrics[primary] ?? 0);

  const ctx = canvas.getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: primary.replace(/_/g, ' '),
        data: vals,
        backgroundColor: vals.map((_, i) => i === 0 ? 'rgba(47,255,180,0.8)' : 'rgba(99,102,241,0.5)'),
        borderColor:     vals.map((_, i) => i === 0 ? '#2fffb4' : '#6366f1'),
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          ticks: { color: '#64748b', font: { family: 'Space Mono', size: 10 } },
          grid:  { color: '#252b34' },
        },
        y: {
          ticks: { color: '#64748b', font: { family: 'Space Mono', size: 10 } },
          grid:  { color: '#252b34' },
        },
      },
    },
  });
}

$('btn-to-predict')?.addEventListener('click', async () => {
  setScreen('predict');
  await loadPredictionForm();
});

$('btn-to-download')?.addEventListener('click', () => {
  setScreen('download');
  loadDownloads();
});

// ─── Screen: PREDICT ──────────────────────────────────────────────────────────
async function loadPredictionForm() {
  try {
    const data = await apiFetch('/api/feature-names');
    state.featureNames = data.features;
    renderPredictForm(data.features);

    // Load history
    const hist = await apiFetch('/api/predictions');
    state.predHistory = hist.history || [];
    renderPredHistory(state.predHistory);
  } catch (err) {
    toast(err.message, 'error');
  }
}

function renderPredictForm(features) {
  const container = $('predict-fields');
  container.innerHTML = features.map(f => `
    <div class="form-group">
      <label class="form-label">${f}</label>
      <input type="number" step="any" class="form-control predict-input" data-feature="${f}" placeholder="0" value="0">
    </div>`).join('');
}

$('btn-run-predict')?.addEventListener('click', async () => {
  const inputs = document.querySelectorAll('.predict-input');
  const features = {};
  inputs.forEach(inp => {
    features[inp.dataset.feature] = parseFloat(inp.value) || 0;
  });

  $('predict-result-area').innerHTML = '<div class="spinner" style="margin:0 auto"></div>';

  try {
    const data = await apiFetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    });

    const r = data.result;
    const displayVal = r.label || r.prediction;
    const conf = r.confidence;

    $('predict-result-area').innerHTML = `
      <div class="predict-result">
        <div class="result-val">${displayVal}</div>
        <div class="result-label">Prediction</div>
        ${conf !== null && conf !== undefined ? `
          <div class="confidence-bar">
            <div class="confidence-track">
              <div class="confidence-fill" style="width:${conf}%"></div>
            </div>
            <div class="text-muted" style="font-size:0.7rem">Confidence: ${conf.toFixed(1)}%</div>
          </div>` : ''}
      </div>`;

    state.predHistory = data.history || [];
    renderPredHistory(state.predHistory);
    toast('Prediction complete ✓', 'success');
  } catch (err) {
    $('predict-result-area').innerHTML = `<div class="alert alert-error">${err.message}</div>`;
    toast(err.message, 'error');
  }
});

function renderPredHistory(history) {
  const el = $('pred-history');
  if (!history.length) {
    el.innerHTML = '<span class="text-muted">No predictions yet</span>';
    return;
  }
  el.innerHTML = [...history].reverse().slice(0, 10).map(p => {
    const ts = p.timestamp ? new Date(p.timestamp).toLocaleTimeString() : '';
    const val = p.label || p.prediction;
    return `<div class="history-item">
      <span class="h-pred">${val}</span>
      ${p.confidence ? `<span>${p.confidence.toFixed(1)}%</span>` : ''}
      <span class="h-time">${ts}</span>
    </div>`;
  }).join('');
}

// ─── Screen: DOWNLOAD ─────────────────────────────────────────────────────────
async function loadDownloads() {
  try {
    const data = await apiFetch('/api/downloads');
    renderDownloads(data.files || []);
  } catch (err) {
    toast(err.message, 'error');
  }
}

const DL_ICONS = {
  features:      '📊',
  model:         '🤖',
  report:        '📋',
  metrics:       '📈',
  test_features: '🧪',
  test_target:   '🎯',
};

function renderDownloads(files) {
  const el = $('download-grid');
  if (!files.length) {
    el.innerHTML = '<span class="text-muted">No files available yet. Complete training first.</span>';
    return;
  }
  el.innerHTML = files.map(f => `
    <div class="download-item">
      <div class="download-icon">${DL_ICONS[f.key] || '📄'}</div>
      <div class="download-label">${f.label}</div>
      <div class="download-filename">${f.filename}</div>
      <a href="/api/download/${f.key}" download="${f.filename}" class="btn btn-outline" style="margin-top:auto">
        ↓ Download
      </a>
    </div>`).join('');
}

// ─── Clear session ────────────────────────────────────────────────────────────
$('btn-clear-session')?.addEventListener('click', async () => {
  if (!confirm('Clear all session data? This cannot be undone.')) return;
  try {
    await apiFetch('/api/session', { method: 'DELETE' });
    state.stage = 'empty';
    state.datasetInfo = null;
    updateSessionBadge();
    setScreen('home');
    hide('restore-banner');
    resetUploadZone();
    hide('dataset-info-section');
    toast('Session cleared', 'info');
  } catch (err) {
    toast(err.message, 'error');
  }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setScreen('home');
  initHome();
  initUpload();
});
