/**
 * dashboard.js
 * Controls the Admin Dashboard functionality:
 * - Dynamic KPI summary metrics
 * - Interactive Chart.js visual charts (Conversion Doughnut & Source Propensity)
 * - Real-time ML Prediction Form submission & Result Rendering
 * - Preset lead scenarios
 * - Dynamic Customer Prediction Table with real-time filtering & search
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const kpiTotal = document.getElementById('kpi-total');
  const kpiConverted = document.getElementById('kpi-converted');
  const kpiNonConverted = document.getElementById('kpi-non-converted');
  const kpiRate = document.getElementById('kpi-rate');

  const predForm = document.getElementById('prediction-form');
  const formSource = document.getElementById('form_source');
  const formService = document.getElementById('form_service');
  const formBudget = document.getElementById('form_budget');
  const formResponseTime = document.getElementById('form_response_time');
  const formFollowUps = document.getElementById('form_follow_ups');
  const btnPredictSubmit = document.getElementById('btn-predict-submit');
  const btnPredictText = document.getElementById('btn-predict-text');
  const predictSpinner = document.getElementById('predict-spinner');
  const predErrorAlert = document.getElementById('pred-error-alert');
  const predErrorText = document.getElementById('pred-error-text');

  // Result Elements
  const predResultPanel = document.getElementById('prediction-result');
  const resBadge = document.getElementById('res-badge');
  const resIcon = document.getElementById('res-icon');
  const resStatusText = document.getElementById('res-status-text');
  const resEnquiryId = document.getElementById('res-enquiry-id');
  const resProba = document.getElementById('res-proba');
  const resBarFill = document.getElementById('res-bar-fill');
  const resPriority = document.getElementById('res-priority');
  const resAction = document.getElementById('res-action');

  // Table Elements
  const customerTbody = document.getElementById('customer-tbody');
  const tableSearch = document.getElementById('table-search');
  const tableCountLabel = document.getElementById('table-count-label');
  const filterButtons = document.querySelectorAll('.filter-btn');

  // Preset Buttons
  const presetButtons = document.querySelectorAll('.preset-btn');

  // In-memory data store
  let customersData = [];
  let currentFilter = 'all';
  let currentSearchQuery = '';

  // Chart references
  let doughnutChart = null;
  let barChart = null;

  // ---------------------------------------------------------------------------
  // 1. DATA INITIALIZATION
  // ---------------------------------------------------------------------------
  async function fetchInitialData() {
    try {
      const [statsRes, custRes] = await Promise.all([
        fetch('/api/stats').then(r => r.json()),
        fetch('/api/customers').then(r => r.json()),
      ]);

      if (custRes.success && custRes.customers) {
        customersData = custRes.customers;
        renderTable();
      }

      if (statsRes.success && statsRes.stats) {
        updateKPIs(statsRes.stats);
        initCharts(statsRes.stats);
      }
    } catch (err) {
      console.error('Failed to load initial dashboard data:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // 2. KPI UPDATES
  // ---------------------------------------------------------------------------
  function updateKPIs(stats) {
    if (!stats) return;
    if (kpiTotal) kpiTotal.textContent = stats.total;
    if (kpiConverted) kpiConverted.textContent = stats.converted;
    if (kpiNonConverted) kpiNonConverted.textContent = stats.non_converted;
    if (kpiRate) kpiRate.textContent = `${stats.rate}%`;
  }

  // ---------------------------------------------------------------------------
  // 3. CHART.JS INITIALIZATION & UPDATES
  // ---------------------------------------------------------------------------
  function initCharts(stats) {
    // A. Doughnut Chart: Conversion Distribution
    const ctxDoughnut = document.getElementById('conversionDoughnutChart');
    if (ctxDoughnut && typeof Chart !== 'undefined') {
      const conv = stats.converted || 0;
      const nonConv = stats.non_converted || 0;

      doughnutChart = new Chart(ctxDoughnut, {
        type: 'doughnut',
        data: {
          labels: ['Converted (Label 1)', 'Not Converted (Label 0)'],
          datasets: [{
            data: [conv, nonConv],
            backgroundColor: ['#10B981', '#EF4444'],
            borderColor: '#111827',
            borderWidth: 3,
            hoverOffset: 6,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '70%',
          plugins: {
            legend: {
              position: 'bottom',
              labels: {
                color: '#CBD5E1',
                padding: 16,
                font: { family: 'Plus Jakarta Sans', size: 12 },
              },
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  const val = context.raw || 0;
                  const total = conv + nonConv;
                  const pct = total > 0 ? ((val / total) * 100).toFixed(1) : 0;
                  return ` ${context.label}: ${val} leads (${pct}%)`;
                },
              },
            },
          },
        },
      });
    }

    // B. Bar Chart: Source Propensity
    const ctxBar = document.getElementById('sourceBarChart');
    if (ctxBar && typeof Chart !== 'undefined' && stats.by_source) {
      const labels = Object.keys(stats.by_source);
      const rates = labels.map(s => stats.by_source[s].rate);

      barChart = new Chart(ctxBar, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Conversion Rate (%)',
            data: rates,
            backgroundColor: '#6366F1',
            borderRadius: 6,
            hoverBackgroundColor: '#818CF8',
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#94A3B8', callback: v => `${v}%` },
            },
            x: {
              grid: { display: false },
              ticks: { color: '#CBD5E1', font: { size: 11 } },
            },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => ` Conversion Rate: ${ctx.raw}%`,
              },
            },
          },
        },
      });
    }
  }

  function updateChartsAfterPrediction(stats) {
    if (doughnutChart) {
      doughnutChart.data.datasets[0].data = [stats.converted, stats.non_converted];
      doughnutChart.update();
    }
    if (barChart && stats.by_source) {
      const labels = Object.keys(stats.by_source);
      barChart.data.datasets[0].data = labels.map(s => stats.by_source[s].rate);
      barChart.update();
    }
  }

  // ---------------------------------------------------------------------------
  // 4. CUSTOMER TABLE RENDERING & FILTERING
  // ---------------------------------------------------------------------------
  function renderTable() {
    let filtered = customersData;

    // Filter by conversion status
    if (currentFilter === 'converted') {
      filtered = filtered.filter(c => c.converted === 1);
    } else if (currentFilter === 'not_converted') {
      filtered = filtered.filter(c => c.converted === 0);
    }

    // Filter by search query
    if (currentSearchQuery) {
      const q = currentSearchQuery.toLowerCase();
      filtered = filtered.filter(c => 
        (c.enquiry_id && c.enquiry_id.toLowerCase().includes(q)) ||
        (c.source && c.source.toLowerCase().includes(q)) ||
        (c.service_type && c.service_type.toLowerCase().includes(q))
      );
    }

    // Update count label
    if (tableCountLabel) {
      tableCountLabel.textContent = `Showing ${filtered.length} of ${customersData.length} records`;
    }

    if (filtered.length === 0) {
      customerTbody.innerHTML = `
        <tr>
          <td colspan="8" style="text-align: center; padding: 2rem; color: #94A3B8;">
            No customer enquiry records matching criteria.
          </td>
        </tr>
      `;
      return;
    }

    const rowsHtml = filtered.map(item => {
      const isConv = item.converted === 1;
      const statusBadge = isConv
        ? `<span class="badge-status badge-status-conv">✓ Converted</span>`
        : `<span class="badge-status badge-status-not">✕ Not Converted</span>`;
      
      const probaText = item.probability ? `${item.probability}%` : (isConv ? '85.0%' : '20.0%');
      const isNew = item.source_type === 'Live Admin Inference';

      return `
        <tr style="${isNew ? 'background: rgba(99, 102, 241, 0.08);' : ''}">
          <td class="mono-cell">${escapeHtml(item.enquiry_id)} ${isNew ? '<span style="font-size: 0.65rem; background: #6366F1; color: white; padding: 1px 5px; border-radius: 4px; margin-left: 4px;">NEW</span>' : ''}</td>
          <td><strong>${escapeHtml(item.source)}</strong></td>
          <td>${escapeHtml(item.service_type)}</td>
          <td>$${Number(item.budget).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td>${item.response_time_hours}h</td>
          <td>${item.previous_interactions_follow_up_count}</td>
          <td>${statusBadge}</td>
          <td style="color: ${isConv ? '#34D399' : '#94A3B8'}; font-weight: 600;">${probaText}</td>
        </tr>
      `;
    }).join('');

    customerTbody.innerHTML = rowsHtml;
  }

  // Filter Buttons Click Handler
  filterButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      filterButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.getAttribute('data-filter');
      renderTable();
    });
  });

  // Search Input Handler
  if (tableSearch) {
    tableSearch.addEventListener('input', (e) => {
      currentSearchQuery = e.target.value.trim();
      renderTable();
    });
  }

  // ---------------------------------------------------------------------------
  // 5. PRESET SCENARIO BUTTONS
  // ---------------------------------------------------------------------------
  presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const preset = btn.getAttribute('data-preset');
      if (preset === 'referral') {
        formSource.value = 'Referral';
        formService.value = 'Enterprise';
        formBudget.value = '28000';
        formResponseTime.value = '1.5';
        formFollowUps.value = '4';
      } else if (preset === 'enterprise') {
        formSource.value = 'Walk-in';
        formService.value = 'Enterprise';
        formBudget.value = '35000';
        formResponseTime.value = '2.0';
        formFollowUps.value = '5';
      } else if (preset === 'cold') {
        formSource.value = 'Social Media';
        formService.value = 'Basic';
        formBudget.value = '1100';
        formResponseTime.value = '42.0';
        formFollowUps.value = '0';
      } else if (preset === 'average') {
        formSource.value = 'Website';
        formService.value = 'Standard';
        formBudget.value = '4800';
        formResponseTime.value = '12.0';
        formFollowUps.value = '2';
      }
      hidePredError();
    });
  });

  // ---------------------------------------------------------------------------
  // 6. ML PREDICTION FORM SUBMISSION
  // ---------------------------------------------------------------------------
  if (predForm) {
    predForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      hidePredError();

      const source = formSource.value;
      const service = formService.value;
      const budget = parseFloat(formBudget.value);
      const respTime = parseFloat(formResponseTime.value);
      const followUps = parseInt(formFollowUps.value, 10);

      // Validation
      if (isNaN(budget) || budget < 0) {
        showPredError('Budget must be a non-negative number.');
        return;
      }
      if (isNaN(respTime) || respTime < 0) {
        showPredError('Response time cannot be negative.');
        return;
      }
      if (isNaN(followUps) || followUps < 0) {
        showPredError('Follow-up count cannot be negative.');
        return;
      }

      // Set Loading State
      btnPredictSubmit.disabled = true;
      btnPredictText.textContent = 'Running ML Inference...';
      predictSpinner.style.display = 'inline-block';

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: JSON.stringify({
            source: source,
            service_type: service,
            budget: budget,
            response_time_hours: respTime,
            previous_interactions_follow_up_count: followUps,
          }),
        });

        const result = await response.json();

        if (response.ok && result.success) {
          // Display Prediction Result Panel
          displayPredictionResult(result);

          // Update Customer Table (Insert at beginning)
          const newRecord = {
            enquiry_id: result.enquiry_id,
            source: source,
            service_type: service,
            budget: budget,
            response_time_hours: respTime,
            previous_interactions_follow_up_count: followUps,
            converted: result.prediction,
            status_label: result.class_label,
            probability: result.probability,
            source_type: 'Live Admin Inference',
          };
          customersData.unshift(newRecord);
          renderTable();

          // Refresh KPI Metrics
          refreshStats();
        } else {
          showPredError(result.message || 'Error running prediction.');
        }
      } catch (err) {
        showPredError('Network connection error while contacting prediction API.');
        console.error('Prediction API error:', err);
      } finally {
        btnPredictSubmit.disabled = false;
        btnPredictText.textContent = '🔮 Predict Conversion Likelihood';
        predictSpinner.style.display = 'none';
      }
    });
  }

  // Display Result Card
  function displayPredictionResult(res) {
    predResultPanel.style.display = 'block';
    const isConv = res.prediction === 1;

    predResultPanel.className = `prediction-result-panel ${isConv ? 'converted' : 'not-converted'}`;
    resBadge.className = `res-pill ${isConv ? 'res-pill-success' : 'res-pill-danger'}`;
    resIcon.textContent = isConv ? '✅' : '⚠️';
    resStatusText.textContent = isConv ? 'CONVERTED (LABEL 1)' : 'NOT CONVERTED (LABEL 0)';
    resEnquiryId.textContent = `Generated ID: ${res.enquiry_id}`;
    resProba.textContent = `${res.probability}%`;

    // Bar Fill
    resBarFill.style.width = `${Math.max(res.probability, 4)}%`;
    if (res.probability >= 50) {
      resBarFill.className = 'probability-bar-fill prob-green';
    } else if (res.probability >= 30) {
      resBarFill.className = 'probability-bar-fill prob-yellow';
    } else {
      resBarFill.className = 'probability-bar-fill prob-red';
    }

    resPriority.textContent = `📋 Strategic Recommendation (${res.priority}):`;
    resAction.textContent = res.recommendation;
  }

  // Refresh Stats & Charts
  async function refreshStats() {
    try {
      const statsRes = await fetch('/api/stats').then(r => r.json());
      if (statsRes.success && statsRes.stats) {
        updateKPIs(statsRes.stats);
        updateChartsAfterPrediction(statsRes.stats);
      }
    } catch (e) {
      console.warn('Could not refresh stats:', e);
    }
  }

  function showPredError(msg) {
    predErrorText.textContent = msg;
    predErrorAlert.className = 'alert-box alert-danger show shake';
    setTimeout(() => predErrorAlert.classList.remove('shake'), 450);
  }

  function hidePredError() {
    predErrorAlert.className = 'alert-box alert-danger';
    predErrorText.textContent = '';
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Kickoff
  fetchInitialData();
});
