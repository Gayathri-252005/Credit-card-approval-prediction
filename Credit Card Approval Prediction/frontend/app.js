/* =============================================================
   CreditIQ — Frontend Application Logic
   API Base URL — change if backend runs on a different port
   ============================================================= */

const API_BASE = 'http://localhost:8000/api/v1';

/* ── State ──────────────────────────────────────────────────── */
let currentTab     = 'analyst';
let wizardStep     = 1;
let totalSteps     = 4;
let batchResults   = [];

/* ── Tab Switching ──────────────────────────────────────────── */
function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');
  document.getElementById(`panel-${tab}`).classList.add('active');
  currentTab = tab;
}

/* ── API Health Check ───────────────────────────────────────── */
async function checkApiHealth() {
  const dot  = document.getElementById('api-status-dot');
  const text = document.getElementById('api-status-text');
  try {
    const res  = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (data.status === 'healthy') {
      dot.className  = 'status-dot online';
      text.textContent = 'API Online';
      document.getElementById('header-model-type').textContent = data.model_type || 'XGBoost';
      document.getElementById('header-accuracy').textContent   =
        data.accuracy ? `Acc: ${(data.accuracy * 100).toFixed(1)}%` : '';
    } else {
      dot.className  = 'status-dot';
      text.textContent = 'Model Not Loaded';
    }
  } catch {
    dot.className  = 'status-dot offline';
    text.textContent = 'API Offline';
  }
}

/* ── Loading Helpers ────────────────────────────────────────── */
function showLoading(msg = 'Processing...') {
  document.getElementById('loading-text').textContent = msg;
  document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
  document.getElementById('loading-overlay').style.display = 'none';
}

/* ── Credit Score Bar ───────────────────────────────────────── */
document.getElementById('a-credit-score')?.addEventListener('input', function () {
  const val  = parseInt(this.value) || 300;
  const pct  = ((val - 300) / 550) * 100;
  document.getElementById('score-bar-fill').style.width = `${pct}%`;
  const labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'];
  const idx = Math.min(Math.floor(pct / 20), 4);
  document.getElementById('score-label').textContent = labels[idx];
});

/* ── ANALYST FORM ───────────────────────────────────────────── */
async function submitAnalystForm(e) {
  e.preventDefault();
  showLoading('Running AI screening...');

  const payload = {
    credit_score:          parseInt(document.getElementById('a-credit-score').value),
    annual_income:         parseFloat(document.getElementById('a-annual-income').value),
    debt_to_income_ratio:  parseFloat(document.getElementById('a-dti').value),
    employment_months:     parseInt(document.getElementById('a-employment').value),
    credit_history_months: parseInt(document.getElementById('a-history').value),
    income_type:           document.getElementById('a-income-type').value,
    payment_status:        parseInt(document.getElementById('a-payment-status').value),
    num_open_accounts:     parseInt(document.getElementById('a-accounts').value),
    monthly_expenses:      parseFloat(document.getElementById('a-expenses').value) || null,
  };

  try {
    const res  = await fetch(`${API_BASE}/predict-single`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderAnalystResult(data);
  } catch (err) {
    alert(`Error: ${err.message}\n\nMake sure the backend is running:\n  cd backend && py -m uvicorn app.main:app --reload`);
  } finally {
    hideLoading();
  }
}

function renderAnalystResult(data) {
  // Hide empty state, show cards
  document.getElementById('analyst-empty').style.display    = 'none';
  document.getElementById('analyst-decision-card').style.display = '';
  document.getElementById('importance-card').style.display  = '';
  document.getElementById('tips-card').style.display        = '';

  // Decision
  const decision  = data.decision;
  const iconEl    = document.getElementById('decision-icon');
  const labelEl   = document.getElementById('decision-label');
  const subEl     = document.getElementById('decision-sub');

  iconEl.className = 'decision-icon';
  labelEl.className = 'decision-label';

  if (decision === 'APPROVED') {
    iconEl.classList.add('approved');
    iconEl.textContent = '✅';
    labelEl.classList.add('approved');
    labelEl.textContent = 'APPROVED';
    subEl.textContent = 'Application cleared for credit product';
  } else if (decision === 'COMPLIANCE_REJECTED') {
    iconEl.classList.add('compliance');
    iconEl.textContent = '⚠️';
    labelEl.classList.add('compliance');
    labelEl.textContent = 'COMPLIANCE REJECTED';
    subEl.textContent = 'Auto-disqualified by compliance rules';
  } else {
    iconEl.classList.add('rejected');
    iconEl.textContent = '❌';
    labelEl.classList.add('rejected');
    labelEl.textContent = 'REJECTED';
    subEl.textContent = 'Application does not meet criteria';
  }

  // Probability ring
  const prob     = data.probability;
  const probPct  = Math.round(prob * 100);
  const circumference = 2 * Math.PI * 34;
  const offset   = circumference * (1 - prob);
  document.getElementById('ring-fill').style.strokeDashoffset = offset;
  document.getElementById('ring-pct').textContent = `${probPct}%`;

  // Add SVG gradient if not present
  const svg = document.getElementById('prob-ring-svg');
  if (!svg.querySelector('defs')) {
    svg.insertAdjacentHTML('afterbegin', `
      <defs>
        <linearGradient id="ringGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#6366F1"/>
          <stop offset="100%" stop-color="#8B5CF6"/>
        </linearGradient>
      </defs>`);
  }

  // Compliance banner
  const banner = document.getElementById('compliance-banner');
  const compText = document.getElementById('compliance-text');
  banner.style.display = '';
  if (data.is_compliance_rejected || data.compliance_status !== 'Compliant') {
    banner.className = 'compliance-banner high-risk';
    compText.textContent = `⚠️ ${data.compliance_status} — Payment history disqualifies applicant`;
  } else {
    banner.className = 'compliance-banner compliant';
    compText.textContent = `✓ ${data.compliance_status} — No adverse payment history`;
  }

  // Risk Score
  const risk = data.risk_score;
  document.getElementById('risk-value').textContent = risk.toFixed(1);
  document.getElementById('risk-meter-fill').style.width = `${risk}%`;
  const tierBadge = document.getElementById('risk-tier-badge');
  tierBadge.textContent = data.risk_tier;
  tierBadge.className = `risk-tier-badge ${data.risk_tier.replace(' ', '-')}`;

  // Feature Importance
  const chartEl = document.getElementById('importance-chart');
  chartEl.innerHTML = '';
  const items = data.feature_importance || [];
  const maxImp = Math.max(...items.map(i => i.importance), 0.001);
  items.forEach(item => {
    const pct = ((item.importance / maxImp) * 100).toFixed(1);
    chartEl.insertAdjacentHTML('beforeend', `
      <div class="importance-bar-item">
        <div class="importance-feat-name" title="${item.feature}">${formatFeatureName(item.feature)}</div>
        <div class="importance-bar-track">
          <div class="importance-bar-fill ${item.direction}" style="width:${pct}%"></div>
        </div>
        <div class="importance-pct">${(item.importance * 100).toFixed(1)}%</div>
      </div>`);
  });

  // Tips
  const tipsList = document.getElementById('tips-list');
  tipsList.innerHTML = '';
  (data.improvement_tips || []).forEach(tip => {
    tipsList.insertAdjacentHTML('beforeend', `<li>${tip}</li>`);
  });
}

function formatFeatureName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
    .replace('Dti', 'DTI')
    .replace('Num ', '# ');
}

/* ── COMPLIANCE BATCH ───────────────────────────────────────── */
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.add('drag-over');
}

function handleDragLeave(e) {
  document.getElementById('dropzone').classList.remove('drag-over');
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) processCSVFile(file);
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) processCSVFile(file);
}

async function processCSVFile(file) {
  if (!file.name.endsWith('.csv')) {
    alert('Please upload a .csv file');
    return;
  }

  showLoading('Processing batch screening...');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/batch-screening`, {
      method: 'POST',
      body:   formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    batchResults = data.results;
    renderBatchResults(data);
  } catch (err) {
    alert(`Error: ${err.message}\n\nMake sure the backend is running.`);
  } finally {
    hideLoading();
  }
}

function renderBatchResults(data) {
  // Stats
  document.getElementById('batch-stats').style.display = '';
  document.getElementById('stat-total').textContent     = data.total_applicants;
  document.getElementById('stat-approved').textContent  = data.approved_count;
  document.getElementById('stat-rejected').textContent  = data.rejected_count;
  document.getElementById('stat-compliance').textContent = data.compliance_rejected_count;
  document.getElementById('stat-highrisk').textContent  = data.high_risk_count;
  document.getElementById('stat-rate').textContent      = `${(data.approval_rate * 100).toFixed(1)}%`;

  // Table
  document.getElementById('batch-table-card').style.display = '';
  document.getElementById('export-btn').style.display       = '';

  const tbody = document.getElementById('batch-tbody');
  tbody.innerHTML = '';

  data.results.forEach(r => {
    const decisionClass = r.decision === 'APPROVED' ? 'approved'
                        : r.decision === 'COMPLIANCE_REJECTED' ? 'compliance' : 'rejected';
    const compClass = r.compliance_status === 'Compliant' ? 'compliant' : 'high-risk';
    const riskClass = `risk-${r.risk_tier.toLowerCase().replace(' ', '-')}`;

    tbody.insertAdjacentHTML('beforeend', `
      <tr>
        <td class="mono">${r.row_index + 1}</td>
        <td class="mono">${r.applicant_id || '—'}</td>
        <td class="mono">${r.credit_score ?? '—'}</td>
        <td class="mono">${r.annual_income ? '$' + r.annual_income.toLocaleString() : '—'}</td>
        <td class="mono">${r.payment_status}</td>
        <td><span class="badge badge-${compClass}">${r.compliance_status}</span></td>
        <td><span class="badge badge-${decisionClass}">${r.decision.replace('_', ' ')}</span></td>
        <td class="mono">${(r.probability * 100).toFixed(1)}%</td>
        <td class="mono">${r.risk_score.toFixed(1)}</td>
        <td><span class="badge badge-${riskClass}">${r.risk_tier}</span></td>
      </tr>`);
  });
}

function exportResults() {
  if (!batchResults.length) return;

  const headers = ['Row','Applicant ID','Credit Score','Annual Income','Payment Status',
                   'Compliance Status','Decision','Probability','Risk Score','Risk Tier'];
  const rows = batchResults.map(r => [
    r.row_index + 1,
    r.applicant_id || '',
    r.credit_score || '',
    r.annual_income || '',
    r.payment_status,
    r.compliance_status,
    r.decision,
    (r.probability * 100).toFixed(2) + '%',
    r.risk_score,
    r.risk_tier,
  ]);

  const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `batch_screening_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadSampleCSV() {
  const sample = [
    'payment_status,credit_score,annual_income,debt_to_income_ratio,employment_months,credit_history_months,income_type,num_open_accounts,monthly_expenses',
    '0,720,75000,0.28,48,84,Salaried,4,1750',
    '1,680,55000,0.35,24,60,Self-Employed,3,1600',
    '2,590,42000,0.55,12,36,Salaried,5,1900',
    '3,510,30000,0.70,6,18,Self-Employed,2,1800',
    '4,450,22000,0.85,3,12,Unemployed,1,1500',
    '0,780,120000,0.18,84,144,Salaried,6,2200',
    '1,640,48000,0.40,18,48,Salaried,3,1400',
    '0,750,95000,0.25,60,96,Self-Employed,5,2000',
  ].join('\n');

  const blob = new Blob([sample], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = 'sample_batch.csv';
  a.click();
  URL.revokeObjectURL(url);
}

/* ── CUSTOMER WIZARD ────────────────────────────────────────── */
function updateSlider(fieldId, value, unit) {
  document.getElementById(fieldId).value = value;
  const display = document.getElementById(`${fieldId}-display`);
  if (display) {
    display.querySelector('.slider-val').textContent = value;
  }
}

function validateWizardStep(step) {
  if (step === 1) {
    const income = parseFloat(document.getElementById('w-income').value);
    if (!income || income <= 0) {
      alert('Please enter a valid annual income.');
      return false;
    }
  }
  if (step === 4) {
    const expenses = parseFloat(document.getElementById('w-expenses').value);
    if (!expenses || expenses < 0) {
      alert('Please enter your monthly expenses.');
      return false;
    }
  }
  return true;
}

async function wizardNext() {
  if (!validateWizardStep(wizardStep)) return;

  if (wizardStep < totalSteps) {
    setWizardStep(wizardStep + 1);
  } else {
    // Final step — submit
    await submitWizard();
  }
}

function wizardBack() {
  if (wizardStep > 1) {
    setWizardStep(wizardStep - 1);
  }
}

function setWizardStep(step) {
  // Hide current panel
  document.getElementById(`wizard-step-${wizardStep}`)?.classList.remove('active');
  document.getElementById('wizard-result')?.classList.remove('active');

  // Mark old step completed
  const oldCircle = document.querySelector(`.wizard-step[data-step="${wizardStep}"]`);
  if (oldCircle) {
    oldCircle.classList.remove('active');
    oldCircle.classList.add('completed');
    oldCircle.querySelector('.step-circle').textContent = '✓';
  }

  wizardStep = step;

  // Show new panel
  const panel = document.getElementById(`wizard-step-${step}`);
  if (panel) panel.classList.add('active');

  // Mark new step active
  const newCircle = document.querySelector(`.wizard-step[data-step="${step}"]`);
  if (newCircle) {
    newCircle.classList.remove('completed');
    newCircle.classList.add('active');
    if (newCircle.querySelector('.step-circle').textContent === '✓') {
      newCircle.querySelector('.step-circle').textContent = step;
    }
  }

  // Navigation buttons
  document.getElementById('wizard-back-btn').style.display = step > 1 ? '' : 'none';
  const nextBtn = document.getElementById('wizard-next-btn');
  nextBtn.textContent = step === totalSteps ? 'Check My Eligibility →' : 'Continue →';
}

async function submitWizard() {
  const income    = parseFloat(document.getElementById('w-income').value);
  const incomeType = document.querySelector('input[name="w-income-type"]:checked')?.value || 'Salaried';
  const employment = parseInt(document.getElementById('w-employment').value);
  const history   = parseInt(document.getElementById('w-history').value);
  const expenses  = parseFloat(document.getElementById('w-expenses').value);

  const payload = {
    annual_income:         income,
    income_type:           incomeType,
    employment_months:     employment,
    credit_history_months: history,
    monthly_expenses:      expenses,
  };

  showLoading('Checking your eligibility...');

  try {
    const res = await fetch(`${API_BASE}/eligibility-check`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderEligibilityResult(data);

    // Show result panel
    document.getElementById(`wizard-step-${wizardStep}`)?.classList.remove('active');
    document.getElementById('wizard-result').classList.add('active');
    document.getElementById('wizard-nav').style.display = 'none';

    // Mark last step completed
    const lastCircle = document.querySelector(`.wizard-step[data-step="${wizardStep}"]`);
    if (lastCircle) {
      lastCircle.classList.remove('active');
      lastCircle.classList.add('completed');
      lastCircle.querySelector('.step-circle').textContent = '✓';
    }
  } catch (err) {
    alert(`Error: ${err.message}\n\nMake sure the backend is running.`);
  } finally {
    hideLoading();
  }
}

function renderEligibilityResult(data) {
  const score = data.pre_qualification_score;
  const circumference = 2 * Math.PI * 60; // r=60
  const offset = circumference * (1 - score / 100);

  const tierColors = {
    Platinum: { bg: 'rgba(229,228,226,0.15)', border: 'rgba(229,228,226,0.4)', text: '#E5E4E2', dot: '#E5E4E2' },
    Premium:  { bg: 'rgba(255,215,0,0.15)',   border: 'rgba(255,215,0,0.4)',   text: '#FFD700', dot: '#FFD700' },
    Standard: { bg: 'rgba(192,192,192,0.15)', border: 'rgba(192,192,192,0.4)', text: '#C0C0C0', dot: '#C0C0C0' },
    Secured:  { bg: 'rgba(205,127,50,0.15)',  border: 'rgba(205,127,50,0.4)',  text: '#CD7F32', dot: '#CD7F32' },
  };

  const tier = data.estimated_tier;
  const tc   = tierColors[tier] || tierColors.Standard;

  const cardsHtml = (data.card_recommendations || []).map((card, idx) => `
    <div class="card-rec-item">
      <div class="card-color-dot" style="background:${card.color || '#6366F1'}"></div>
      <div class="card-rec-info">
        <div class="card-rec-name">${card.name}</div>
        <div class="card-rec-details">Limit: ${card.limit} &nbsp;·&nbsp; APR: ${card.apr} &nbsp;·&nbsp; ${card.rewards}</div>
      </div>
      ${idx === 0 ? '<span class="card-rec-best">Best Match</span>' : ''}
    </div>`).join('');

  const tipsHtml = (data.improvement_tips || []).map(tip =>
    `<li>${tip}</li>`).join('');

  const likelihoodColors = {
    'Very Likely': '#10B981',
    'Likely':      '#6366F1',
    'Possible':    '#F59E0B',
    'Unlikely':    '#EF4444',
  };
  const lColor = likelihoodColors[data.likelihood_label] || '#6366F1';

  document.getElementById('eligibility-result').innerHTML = `
    <defs-svg>
      <svg width="0" height="0">
        <defs>
          <linearGradient id="scoreGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="${lColor}"/>
            <stop offset="100%" stop-color="#8B5CF6"/>
          </linearGradient>
        </defs>
      </svg>
    </defs-svg>

    <div class="eligibility-score-wrap">
      <svg class="score-ring-svg" viewBox="0 0 160 160">
        <defs>
          <linearGradient id="scoreGrad2" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="${lColor}"/>
            <stop offset="100%" stop-color="#8B5CF6"/>
          </linearGradient>
        </defs>
        <circle cx="80" cy="80" r="60" class="score-ring-bg"/>
        <circle cx="80" cy="80" r="60" class="score-ring-fill"
          style="stroke:url(#scoreGrad2); stroke-dasharray:${circumference}; stroke-dashoffset:${offset}"/>
      </svg>
      <div class="score-ring-text">
        <div class="score-pct" style="background:linear-gradient(135deg,${lColor},#8B5CF6);-webkit-background-clip:text;-webkit-text-fill-color:transparent">${score.toFixed(0)}</div>
        <div class="score-pct-label">/ 100</div>
      </div>
    </div>

    <div class="eligibility-tier-badge"
      style="background:${tc.bg};border:1px solid ${tc.border};color:${tc.text}">
      ${tier} Tier
    </div>
    <div class="eligibility-likelihood">
      <strong style="color:${lColor}">${data.likelihood_label}</strong> — Likelihood of Approval
    </div>

    <div class="card-recommendations">
      <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);margin-bottom:8px;text-align:left">
        Card Recommendations
      </div>
      ${cardsHtml}
    </div>

    ${tipsHtml ? `
    <div class="eligibility-tips tips-list">
      <h4>How to Improve Your Odds</h4>
      <ul class="tips-list">${tipsHtml}</ul>
    </div>` : ''}

    <div class="soft-pull-note">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      </svg>
      ${data.soft_pull_note}
    </div>

    <button class="btn btn-outline retry-btn" onclick="restartWizard()">
      ↩ Start Over
    </button>
  `;
}

function restartWizard() {
  // Reset all steps
  wizardStep = 1;
  document.querySelectorAll('.wizard-step').forEach((s, i) => {
    s.classList.remove('active', 'completed');
    s.querySelector('.step-circle').textContent = i + 1;
  });
  document.querySelectorAll('.wizard-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('wizard-step-1').classList.add('active');
  document.querySelector('.wizard-step[data-step="1"]').classList.add('active');
  document.getElementById('wizard-nav').style.display = '';
  document.getElementById('wizard-back-btn').style.display = 'none';
  document.getElementById('wizard-next-btn').textContent = 'Continue →';
  // Clear inputs
  document.getElementById('w-income').value = '';
  document.getElementById('w-expenses').value = '';
}

/* ── Init ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  // Poll health every 30s
  setInterval(checkApiHealth, 30000);
});
