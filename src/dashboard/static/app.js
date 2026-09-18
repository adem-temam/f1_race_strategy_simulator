// Phase 6 Interactive Strategy Dashboard Client

let currentCircuit = "bahrain";
let circuitsData = [];
let simChart = null;
let paretoChart = null;
let mcChart = null;
let sensChart = null;

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupCircuitSelector();
  setupSimulatorControls();
  setupOptimizerControls();
  setupMonteCarloControls();
  setupSensitivityControls();
  await loadCircuits();
  runSimulation();
});

// -------------------------------------------------------------------------
// Navigation Tabs
// -------------------------------------------------------------------------
function setupTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }

      // Auto-trigger tab-specific loads
      if (targetId === "validationTab") {
        loadHistoricalValidation();
      } else if (targetId === "scenariosTab") {
        loadScenarioPresets();
      }
    });
  });
}

// -------------------------------------------------------------------------
// Circuit Selector
// -------------------------------------------------------------------------
async function loadCircuits() {
  try {
    const res = await fetch("/api/circuits");
    circuitsData = await res.json();
    const select = document.getElementById("circuitSelect");
    select.innerHTML = "";
    circuitsData.forEach(c => {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = c.name;
      if (c.id === currentCircuit) opt.selected = true;
      select.appendChild(opt);
    });
    updateCircuitBadges();
  } catch (err) {
    console.error("Failed to load circuits:", err);
  }
}

function setupCircuitSelector() {
  const select = document.getElementById("circuitSelect");
  select.addEventListener("change", (e) => {
    currentCircuit = e.target.value;
    updateCircuitBadges();
    // Refresh active tab
    const activeTab = document.querySelector(".tab-btn.active")?.getAttribute("data-tab");
    if (activeTab === "validationTab") loadHistoricalValidation();
    if (activeTab === "scenariosTab") loadScenarioPresets();
  });
}

function updateCircuitBadges() {
  const c = circuitsData.find(item => item.id === currentCircuit);
  if (!c) return;
  document.getElementById("circuitLapsBadge").textContent = `${c.total_laps} Laps`;
  document.getElementById("circuitBasePaceBadge").textContent = `Base: ${c.base_lap_time}s`;

  const pitBadge = document.getElementById("circuitPitLossBadge");
  if (pitBadge && c.pit_loss) pitBadge.textContent = `Pit: ${c.pit_loss.toFixed(1)}s`;

  const degBadge = document.getElementById("circuitDegBadge");
  if (degBadge && c.tyre_degradation_multiplier) {
    degBadge.textContent = `Deg: ${c.tyre_degradation_multiplier.toFixed(2)}x`;
  }

  // Sync sliders to circuit baseline
  const degSlider = document.getElementById("simDegSlider");
  const degVal = document.getElementById("simDegVal");
  if (degSlider && degVal) {
    degSlider.value = c.tyre_degradation_multiplier || 1.0;
    degVal.textContent = `${parseFloat(degSlider.value).toFixed(2)}x`;
  }

  const pitSlider = document.getElementById("simPitSlider");
  const pitVal = document.getElementById("simPitVal");
  if (pitSlider && pitVal) {
    pitSlider.value = c.pit_loss || 22.0;
    pitVal.textContent = `${parseFloat(pitSlider.value).toFixed(1)}s`;
  }

  // Populate valid default strategies for this circuit
  if (c.default_strategies && c.default_strategies.length > 0) {
    const container = document.getElementById("strategyInputsContainer");
    if (container) {
      container.innerHTML = "";
      c.default_strategies.forEach(s => {
        const input = document.createElement("input");
        input.type = "text";
        input.className = "strategy-input-field";
        input.value = s;
        input.style.marginBottom = "0.4rem";
        container.appendChild(input);
      });
    }
  }
}

// -------------------------------------------------------------------------
// Tab 1: Race Simulator
// -------------------------------------------------------------------------
function setupSimulatorControls() {
  // Sliders display sync
  const degSlider = document.getElementById("simDegSlider");
  const degVal = document.getElementById("simDegVal");
  degSlider.addEventListener("input", () => {
    degVal.textContent = `${parseFloat(degSlider.value).toFixed(2)}x`;
  });

  const pitSlider = document.getElementById("simPitSlider");
  const pitVal = document.getElementById("simPitVal");
  pitSlider.addEventListener("input", () => {
    pitVal.textContent = `${parseFloat(pitSlider.value).toFixed(1)}s`;
  });

  // Add strategy input
  document.getElementById("addStrategyInputBtn").addEventListener("click", () => {
    const container = document.getElementById("strategyInputsContainer");
    const input = document.createElement("input");
    input.type = "text";
    input.className = "strategy-input-field";
    input.value = "S12-H45";
    input.style.marginBottom = "0.4rem";
    container.appendChild(input);
  });

  // Run simulation
  document.getElementById("runSimulationBtn").addEventListener("click", runSimulation);
}

async function runSimulation() {
  const inputs = document.querySelectorAll(".strategy-input-field");
  const strategies = Array.from(inputs).map(i => i.value.trim()).filter(v => v.length > 0);

  const degMult = parseFloat(document.getElementById("simDegSlider").value);
  const pitLoss = parseFloat(document.getElementById("simPitSlider").value);

  const btn = document.getElementById("runSimulationBtn");
  btn.innerHTML = `<span class="spinner"></span> Simulating...`;

  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        circuit: currentCircuit,
        strategies: strategies,
        deg_multiplier: degMult,
        pit_loss: pitLoss,
      }),
    });

    const data = await res.json();
    renderSimulationResults(data);
  } catch (err) {
    alert("Simulation failed: " + err);
  } finally {
    btn.innerHTML = `<span>Simulate Strategies</span>`;
  }
}

function renderSimulationResults(data) {
  const tbody = document.getElementById("simResultsBody");
  tbody.innerHTML = "";

  const chartDatasets = [];
  const palette = ["#ef4444", "#38bdf8", "#f59e0b", "#10b981", "#a855f7"];

  data.results.forEach((r, idx) => {
    if (r.error) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="6" style="color: #ef4444;">${r.strategy}: ${r.error}</td>`;
      tbody.appendChild(tr);
      return;
    }

    const tr = document.createElement("tr");
    const posTag = idx === 0 ? `<span class="tag tag-p1">P1</span>` : `P${idx + 1}`;
    const deltaStr = idx === 0 ? "LEADER" : `+${r.delta.toFixed(3)}s`;

    tr.innerHTML = `
      <td>${posTag}</td>
      <td style="font-weight: 700;">${r.strategy}</td>
      <td>${r.stops} Stop${r.stops !== 1 ? 's' : ''}</td>
      <td class="mono">${r.stints}</td>
      <td class="mono">${r.formatted_time}</td>
      <td class="mono" style="color: ${idx === 0 ? '#22c55e' : '#f59e0b'};">${deltaStr}</td>
    `;
    tbody.appendChild(tr);

    // Chart dataset
    const color = palette[idx % palette.length];
    chartDatasets.push({
      label: r.strategy,
      data: r.laps.map(l => ({ x: l.lap, y: l.effective_lap_time })),
      borderColor: color,
      backgroundColor: color,
      borderWidth: 2,
      pointRadius: r.laps.map(l => l.is_pit_lap ? 5 : 0),
      tension: 0.1,
    });
  });

  // Render chart
  const ctx = document.getElementById("simPaceChart").getContext("2d");
  if (simChart) simChart.destroy();

  simChart = new Chart(ctx, {
    type: "line",
    data: { datasets: chartDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Lap Number", color: "#8c8c9e" },
          grid: { color: "#232330" },
          ticks: { color: "#8c8c9e" },
        },
        y: {
          title: { display: true, text: "Effective Lap Time (s)", color: "#8c8c9e" },
          grid: { color: "#232330" },
          ticks: { color: "#8c8c9e" },
        },
      },
      plugins: {
        legend: { labels: { color: "#f0f0f5" } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.y.toFixed(3)}s (Lap ${ctx.raw.x})`,
          },
        },
      },
    },
  });
}

// -------------------------------------------------------------------------
// Tab 2: Strategy Optimizer
// -------------------------------------------------------------------------
function setupOptimizerControls() {
  document.getElementById("runOptimizerBtn").addEventListener("click", runOptimizer);
}

async function runOptimizer() {
  const optVal = document.getElementById("optMaxStops").value;
  let maxStops = 3;
  let exactStops = null;

  if (optVal === "exact_1") {
    maxStops = 1;
    exactStops = 1;
  } else if (optVal === "exact_2") {
    maxStops = 2;
    exactStops = 2;
  } else if (optVal === "exact_3") {
    maxStops = 3;
    exactStops = 3;
  } else {
    maxStops = 3;
    exactStops = null;
  }

  const objective = document.getElementById("optObjective").value;

  const btn = document.getElementById("runOptimizerBtn");
  btn.innerHTML = `<span class="spinner"></span> Computing DAG Global Minimum...`;

  try {
    const res = await fetch("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        circuit: currentCircuit,
        max_stops: maxStops,
        exact_stops: exactStops,
        objective: objective,
        mc_iterations: 150,
      }),
    });

    const data = await res.json();
    renderOptimizerResults(data);
  } catch (err) {
    alert("Optimization failed: " + err);
  } finally {
    btn.innerHTML = `<span>Discover Optimal Strategy</span>`;
  }
}

function renderOptimizerResults(data) {
  // 1. Optimal card
  const card = document.getElementById("optimalStrategyCard");
  card.style.display = "block";
  const stopsCount = data.optimal_stops;
  const tyresCount = stopsCount + 1;
  const labelSuffix = ` [${stopsCount} Stop${stopsCount !== 1 ? 's' : ''} / ${tyresCount} Tyres]`;
  document.getElementById("optimalStrategyName").textContent = data.optimal_strategy + labelSuffix;
  document.getElementById("optimalStrategyTime").textContent = `Total Time: ${data.formatted_optimal_time}`;
  document.getElementById("optimalSolveMetrics").textContent = 
    `Evaluated ${data.evaluations_count.toLocaleString()} strategies in ${data.solve_time_seconds.toFixed(3)}s`;

  // 2. Tactical pit windows
  const pwContainer = document.getElementById("pitWindowsContainer");
  pwContainer.innerHTML = "";
  if (data.pit_windows.length === 0) {
    pwContainer.innerHTML = `<p style="color: var(--text-muted); font-size: 0.85rem;">Zero-stop strategy has no pit windows.</p>`;
  } else {
    data.pit_windows.forEach(pw => {
      const div = document.createElement("div");
      div.className = "card";
      div.style.padding = "0.85rem 1.15rem";
      div.style.flex = "1";
      div.style.minWidth = "200px";
      div.innerHTML = `
        <div style="font-size: 0.75rem; color: var(--accent-cyan); font-weight: 700;">STOP ${pw.pit_index} (${pw.compound_before} ➔ ${pw.compound_after})</div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #fff; margin: 0.2rem 0;">Lap ${pw.optimal_lap}</div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">Tactical Window: Laps ${pw.window_open_lap}–${pw.window_close_lap} (±${Math.floor(pw.window_size/2)}L)</div>
      `;
      pwContainer.appendChild(div);
    });
  }

  // 3. Leaderboard
  const tbody = document.getElementById("optLeaderboardBody");
  tbody.innerHTML = "";
  data.leaderboard.forEach(row => {
    const tr = document.createElement("tr");
    const deltaStr = row.rank === 1 ? "LEADER" : `+${row.delta_to_p1.toFixed(3)}s`;
    tr.innerHTML = `
      <td>${row.rank === 1 ? '<span class="tag tag-p1">P1</span>' : 'P' + row.rank}</td>
      <td style="font-weight: 700;">${row.description}</td>
      <td>${row.stops} Stop${row.stops !== 1 ? 's' : ''}</td>
      <td class="mono">${row.formatted_time}</td>
      <td class="mono" style="color: ${row.rank === 1 ? '#22c55e' : '#f59e0b'};">${deltaStr}</td>
    `;
    tbody.appendChild(tr);
  });

  // 4. Pareto Frontier Chart
  renderParetoChart(data.pareto_frontier);
}

function renderParetoChart(points) {
  const ctx = document.getElementById("optParetoChart").getContext("2d");
  if (paretoChart) paretoChart.destroy();

  if (!points || points.length === 0) return;

  paretoChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [{
        label: "Pareto Efficient Strategies",
        data: points.map(p => ({ x: p.expected_time, y: p.p95_time, name: p.strategy })),
        backgroundColor: "#06b6d4",
        borderColor: "#38bdf8",
        pointRadius: 6,
        showLine: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: "Expected Total Time (s)", color: "#8c8c9e" },
          grid: { color: "#232330" },
          ticks: { color: "#8c8c9e" },
        },
        y: {
          title: { display: true, text: "P95 Value at Risk (s)", color: "#8c8c9e" },
          grid: { color: "#232330" },
          ticks: { color: "#8c8c9e" },
        },
      },
      plugins: {
        legend: { labels: { color: "#f0f0f5" } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.raw.name}: Pace ${ctx.raw.x.toFixed(1)}s | Risk ${ctx.raw.y.toFixed(1)}s`,
          },
        },
      },
    },
  });
}

// -------------------------------------------------------------------------
// Tab 3: Monte Carlo Explorer
// -------------------------------------------------------------------------
function setupMonteCarloControls() {
  const sync = (sliderId, valId, fmt) => {
    const s = document.getElementById(sliderId);
    const v = document.getElementById(valId);
    s.addEventListener("input", () => { v.textContent = fmt(s.value); });
  };
  sync("mcIterSlider", "mcIterVal", v => parseInt(v).toLocaleString());
  sync("mcLapNoiseSlider", "mcLapNoiseVal", v => `${parseFloat(v).toFixed(2)}s`);
  sync("mcPitSigmaSlider", "mcPitSigmaVal", v => parseFloat(v).toFixed(2));
  sync("mcDegStdSlider", "mcDegStdVal", v => `${Math.round(v * 100)}%`);

  document.getElementById("runMonteCarloBtn").addEventListener("click", runMonteCarlo);
}

async function runMonteCarlo() {
  const inputs = document.querySelectorAll(".strategy-input-field");
  const strategies = Array.from(inputs).map(i => i.value.trim()).filter(v => v.length > 0);

  const iterations = parseInt(document.getElementById("mcIterSlider").value);
  const lapNoise = parseFloat(document.getElementById("mcLapNoiseSlider").value);
  const pitSigma = parseFloat(document.getElementById("mcPitSigmaSlider").value);
  const degStd = parseFloat(document.getElementById("mcDegStdSlider").value);

  const btn = document.getElementById("runMonteCarloBtn");
  btn.innerHTML = `<span class="spinner"></span> Simulating ${iterations.toLocaleString()} Runs...`;

  try {
    const res = await fetch("/api/montecarlo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        circuit: currentCircuit,
        strategies: strategies.length > 0 ? strategies : ["S15-M21-S21", "M26-H31"],
        iterations: iterations,
        lap_noise_std: lapNoise,
        pit_stop_sigma: pitSigma,
        tyre_deg_std: degStd,
      }),
    });

    const data = await res.json();
    renderMonteCarloResults(data);
  } catch (err) {
    alert("Monte Carlo simulation failed: " + err);
  } finally {
    btn.innerHTML = `<span>Simulate Stochastic Outcomes</span>`;
  }
}

function renderMonteCarloResults(data) {
  // 1. Stats Table
  const tbody = document.getElementById("mcStatsBody");
  tbody.innerHTML = "";

  const chartDatasets = [];
  const palette = ["#ef4444", "#38bdf8", "#f59e0b", "#10b981", "#a855f7"];

  data.distributions.forEach((d, idx) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-weight: 700;">${d.strategy}</td>
      <td class="mono">${(d.mean_time / 60.0).toFixed(2)}m</td>
      <td class="mono">${d.std_dev.toFixed(2)}s</td>
      <td class="mono" style="color: #f59e0b;">${(d.p95_time / 60.0).toFixed(2)}m</td>
    `;
    tbody.appendChild(tr);

    // Chart density curve
    const color = palette[idx % palette.length];
    chartDatasets.push({
      label: d.strategy,
      data: d.hist_bins.map((bin, bIdx) => ({ x: bin / 60.0, y: d.hist_density[bIdx] })),
      borderColor: color,
      backgroundColor: color + "20",
      fill: true,
      tension: 0.3,
      borderWidth: 2,
    });
  });

  // 2. Win Matrix Table
  const winBody = document.getElementById("mcWinMatrixBody");
  winBody.innerHTML = "";
  const names = data.win_matrix.strategies;
  const probs = data.win_matrix.probabilities;

  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const p = (probs[i][j] * 100).toFixed(1);
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-size: 0.82rem;"><strong>${names[i]}</strong> beats <strong>${names[j]}</strong></td>
        <td class="mono" style="font-weight: 800; color: ${p >= 50 ? '#22c55e' : '#ef4444'};">${p}%</td>
      `;
      winBody.appendChild(tr);
    }
  }

  // 3. Density Chart
  const ctx = document.getElementById("mcDensityChart").getContext("2d");
  if (mcChart) mcChart.destroy();

  mcChart = new Chart(ctx, {
    type: "line",
    data: { datasets: chartDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Total Race Time (Minutes)", color: "#8c8c9e" },
          grid: { color: "#232330" },
          ticks: { color: "#8c8c9e" },
        },
        y: {
          title: { display: true, text: "Probability Density", color: "#8c8c9e" },
          grid: { color: "#232330" },
          ticks: { color: "#8c8c9e" },
        },
      },
      plugins: {
        legend: { labels: { color: "#f0f0f5" } },
      },
    },
  });
}

// -------------------------------------------------------------------------
// Tab 4: Sensitivity & Phase Maps
// -------------------------------------------------------------------------
function setupSensitivityControls() {
  document.getElementById("runSensitivityBtn").addEventListener("click", runSensitivity1D);
  document.getElementById("runPhaseMapBtn").addEventListener("click", runSensitivity2D);
}

async function runSensitivity1D() {
  const param = document.getElementById("sensParamSelect").value;
  const rMin = parseFloat(document.getElementById("sensRangeMin").value);
  const rMax = parseFloat(document.getElementById("sensRangeMax").value);

  const btn = document.getElementById("runSensitivityBtn");
  btn.innerHTML = `<span class="spinner"></span> Sweeping Parameter...`;

  try {
    const res = await fetch("/api/sensitivity/1d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        circuit: currentCircuit,
        parameter: param,
        range_min: rMin,
        range_max: rMax,
        num_points: 25,
      }),
    });

    const data = await res.json();
    renderSensitivity1D(data);
  } catch (err) {
    alert("Sensitivity sweep failed: " + err);
  } finally {
    btn.innerHTML = `<span>Run Parameter Sweep</span>`;
  }
}

function renderSensitivity1D(data) {
  // 1. Crossover Card
  const card = document.getElementById("crossoverPointCard");
  if (data.crossovers && data.crossovers.length > 0) {
    card.style.display = "block";
    const co = data.crossovers[0];
    document.getElementById("crossoverVal").textContent = `Critical θ* = ${co.crossover_value.toFixed(4)}`;
    document.getElementById("crossoverMsg").textContent = 
      `${co.message} (Nominal: ${co.nominal_value.toFixed(2)}, Distance: ${co.distance_from_nominal > 0 ? '+' : ''}${co.distance_from_nominal.toFixed(4)})`;
  } else {
    card.style.display = "none";
  }

  // 2. Line Chart
  const ctx = document.getElementById("sens1DChart").getContext("2d");
  if (sensChart) sensChart.destroy();

  const chartDatasets = [];
  const palette = ["#ef4444", "#38bdf8", "#f59e0b", "#10b981"];
  let idx = 0;

  for (const [sName, times] of Object.entries(data.curves)) {
    const color = palette[idx % palette.length];
    chartDatasets.push({
      label: sName,
      data: data.parameter_values.map((x, i) => ({ x: x, y: times[i] / 60.0 })),
      borderColor: color,
      borderWidth: 2,
      pointRadius: 0,
    });
    idx++;
  }

  // Add lower envelope
  chartDatasets.push({
    label: "Optimal Lower Envelope",
    data: data.parameter_values.map((x, i) => ({ x: x, y: data.optimal_envelope[i] / 60.0 })),
    borderColor: "#ffffff",
    borderWidth: 3,
    borderDash: [5, 5],
    pointRadius: 0,
  });

  sensChart = new Chart(ctx, {
    type: "line",
    data: { datasets: chartDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: data.display_name, color: "#8c8c9e" },
          grid: { color: "#232330" },
          ticks: { color: "#8c8c9e" },
        },
        y: {
          title: { display: true, text: "Total Race Time (Minutes)", color: "#8c8c9e" },
          grid: { color: "#232330" },
          ticks: { color: "#8c8c9e" },
        },
      },
      plugins: {
        legend: { labels: { color: "#f0f0f5" } },
      },
    },
  });
}

async function runSensitivity2D() {
  const btn = document.getElementById("runPhaseMapBtn");
  btn.innerHTML = `<span class="spinner"></span> Mapping 2D Space...`;

  try {
    const res = await fetch("/api/sensitivity/2d", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        circuit: currentCircuit,
        x_min: 16.0,
        x_max: 30.0,
        y_min: 0.6,
        y_max: 2.0,
        grid_resolution: 20,
      }),
    });

    const data = await res.json();
    drawPhaseMapCanvas(data);
  } catch (err) {
    alert("Phase map failed: " + err);
  } finally {
    btn.innerHTML = `Compute Phase Map`;
  }
}

function drawPhaseMapCanvas(data) {
  const canvas = document.getElementById("phaseMapCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  const grid = data.optimal_stops_grid;
  const numRows = grid.length;
  const numCols = grid[0].length;
  const cellW = w / numCols;
  const cellH = h / numRows;

  // Draw grid regions
  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < numCols; c++) {
      const stops = grid[numRows - 1 - r][c]; // Invert Y
      ctx.fillStyle = stops === 1 ? "rgba(16, 185, 129, 0.4)" : "rgba(225, 6, 0, 0.4)";
      ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  // Draw axis boundaries
  ctx.strokeStyle = "#3e3e50";
  ctx.lineWidth = 1;
  ctx.strokeRect(0, 0, w, h);

  // Plot Nominal Point
  const minX = data.x_values[0];
  const maxX = data.x_values[data.x_values.length - 1];
  const minY = data.y_values[0];
  const maxY = data.y_values[data.y_values.length - 1];

  const normX = (data.nominal_x - minX) / (maxX - minX);
  const normY = (data.nominal_y - minY) / (maxY - minY);

  const ptX = normX * w;
  const ptY = (1.0 - normY) * h;

  ctx.beginPath();
  ctx.arc(ptX, ptY, 7, 0, 2 * Math.PI);
  ctx.fillStyle = "#38bdf8";
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ffffff";
  ctx.stroke();

  // Label
  ctx.font = "bold 11px Inter, sans-serif";
  ctx.fillStyle = "#ffffff";
  ctx.fillText(`Nominal (${data.nominal_stops}S)`, ptX + 10, ptY + 4);
}

// -------------------------------------------------------------------------
// Tab 5: Historical Validation
// -------------------------------------------------------------------------
async function loadHistoricalValidation() {
  try {
    const res = await fetch(`/api/historical/${currentCircuit}`);
    const data = await res.json();

    document.getElementById("valPhysicsError").textContent = `${data.physics_duration_error_pct.toFixed(2)}%`;
    document.getElementById("valCleanRMSE").textContent = `${data.clean_lap_rmse.toFixed(2)}s`;
    document.getElementById("valStopsMatched").textContent = data.stops_matched ? "MATCHED" : "DIVERGED";
    document.getElementById("valStopsMatched").style.color = data.stops_matched ? "#22c55e" : "#f59e0b";

    document.getElementById("valActualStrat").textContent = `${data.driver_name}: ${data.actual_strategy}`;
    document.getElementById("valSimStrat").textContent = `Optimizer: ${data.simulated_optimal_strategy}`;

    // Fitted parameters table
    const tbody = document.getElementById("valParamsBody");
    tbody.innerHTML = "";
    data.fitted_compounds.forEach(fp => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight: 700;">${fp.compound}</td>
        <td class="mono">${fp.alpha.toFixed(4)}</td>
        <td class="mono">${fp.beta.toFixed(5)}</td>
        <td class="mono">${fp.base_offset > 0 ? '+' : ''}${fp.base_offset.toFixed(2)}s</td>
        <td class="mono">${fp.sample_count}</td>
      `;
      tbody.appendChild(tr);
    });

    // Discrepancy Diagnostics
    const diagContainer = document.getElementById("diagnosticsList");
    diagContainer.innerHTML = "";
    if (data.diagnostics.length === 0) {
      diagContainer.innerHTML = `<p style="font-size: 0.8rem; color: #22c55e;">Zero discrepancies detected. High fidelity match.</p>`;
    } else {
      data.diagnostics.forEach(diag => {
        const div = document.createElement("div");
        div.style.background = "#191924";
        div.style.border = "1px solid #333348";
        div.style.borderRadius = "6px";
        div.style.padding = "0.75rem 1rem";
        div.style.marginBottom = "0.6rem";
        div.innerHTML = `
          <div style="font-weight: 700; color: #f59e0b; font-size: 0.85rem;">${diag.title} (~${diag.time_penalty_estimate.toFixed(1)}s impact)</div>
          <div style="font-size: 0.78rem; color: #a1a1aa; margin-top: 0.2rem;">${diag.description}</div>
        `;
        diagContainer.appendChild(div);
      });
    }
  } catch (err) {
    console.error("Historical validation load failed:", err);
  }
}

// -------------------------------------------------------------------------
// Tab 6: What-If Scenarios
// -------------------------------------------------------------------------
async function loadScenarioPresets() {
  try {
    const res = await fetch("/api/scenarios/presets");
    const presets = await res.json();
    renderScenarioPresets(presets);
  } catch (err) {
    console.error("Failed to load scenario presets:", err);
  }
}

function renderScenarioPresets(presets) {
  const container = document.getElementById("scenarioCardsGrid");
  container.innerHTML = "";

  presets.forEach(s => {
    const btn = document.createElement("button");
    btn.className = "scenario-pill-btn";
    btn.innerHTML = `
      <div class="scenario-header">${s.name}</div>
      <div class="scenario-desc">${s.description}</div>
      <div class="scenario-q">"${s.research_question}"</div>
    `;
    btn.addEventListener("click", () => {
      document.querySelectorAll(".scenario-pill-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      triggerScenarioRun(s.id);
    });
    container.appendChild(btn);
  });
}

async function triggerScenarioRun(scenarioId) {
  try {
    const res = await fetch("/api/scenarios/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        circuit: currentCircuit,
        scenario_id: scenarioId,
      }),
    });

    const data = await res.json();
    renderScenarioRunResults(data);
  } catch (err) {
    alert("Scenario run failed: " + err);
  }
}

function renderScenarioRunResults(data) {
  document.getElementById("scenarioOutcomeSection").style.display = "block";

  // Regret banner
  const banner = document.getElementById("regretBanner");
  const title = document.getElementById("regretTitle");
  const desc = document.getElementById("regretDesc");

  if (data.strategic_regret > 0.05) {
    banner.className = "regret-alert pivoted";
    title.textContent = `STRATEGY PIVOT REQUIRED: ${data.formatted_regret} REGRET`;
    desc.textContent = `Adhering to baseline plan costs ${data.formatted_regret} relative to optimal adaptation.`;
  } else {
    banner.className = "regret-alert robust";
    title.textContent = `STRATEGY FULLY ROBUST (0.00s REGRET)`;
    desc.textContent = `Nominal strategy remains optimal under this scenario perturbation.`;
  }

  // Pivot Table
  document.getElementById("scenBaseStrat").textContent = data.baseline_optimal_strategy;
  document.getElementById("scenBaseTime").textContent = data.formatted_baseline_time;
  document.getElementById("scenWinnerStrat").textContent = data.scenario_optimal_strategy;
  document.getElementById("scenWinnerTime").textContent = data.formatted_scenario_time;
  document.getElementById("scenRegretVal").textContent = data.formatted_regret;
  document.getElementById("scenDiagnosticText").textContent = data.summary_insight;

  // Impact Table
  const tbody = document.getElementById("scenarioOutcomesBody");
  tbody.innerHTML = "";
  data.strategy_outcomes.forEach(oc => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-weight: 700;">${oc.strategy_name}</td>
      <td>${oc.stops} Stop${oc.stops !== 1 ? 's' : ''}</td>
      <td class="mono">${oc.formatted_scenario_time}</td>
      <td class="mono" style="color: ${oc.time_impact > 0 ? '#ef4444' : '#22c55e'};">${oc.formatted_impact}</td>
      <td class="mono" style="color: ${oc.gap_to_winner === 0 ? '#22c55e' : '#f59e0b'};">
        ${oc.gap_to_winner === 0 ? 'WINNER' : '+' + oc.gap_to_winner.toFixed(2) + 's'}
      </td>
    `;
    tbody.appendChild(tr);
  });
}
