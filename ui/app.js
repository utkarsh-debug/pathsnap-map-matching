const API_STORAGE_KEY = "pathsnap.apiBase";
const SVG_NS = "http://www.w3.org/2000/svg";

const elements = {
  apiBase: document.getElementById("api-base"),
  connectForm: document.getElementById("connect-form"),
  connectButton: document.getElementById("connect-button"),
  connectionState: document.getElementById("connection-state"),
  connectionLabel: document.getElementById("connection-label"),
  connectionHelp: document.getElementById("connection-help"),
  tripSelect: document.getElementById("trip-select"),
  tripCount: document.getElementById("trip-count"),
  runButton: document.getElementById("run-button"),
  requestNote: document.getElementById("request-note"),
  systemModel: document.getElementById("system-model"),
  systemDecoder: document.getElementById("system-decoder"),
  emptyState: document.getElementById("empty-state"),
  resultsContent: document.getElementById("results-content"),
  resultTripId: document.getElementById("result-trip-id"),
  resultModel: document.getElementById("result-model"),
  mapSummary: document.getElementById("map-summary"),
  mapCaption: document.getElementById("map-caption"),
  routeMap: document.getElementById("route-map"),
  mapGrid: document.getElementById("map-grid"),
  gpsTrace: document.getElementById("gps-trace"),
  gpsPoints: document.getElementById("gps-points"),
  predictedPath: document.getElementById("predicted-path"),
  routeMarkers: document.getElementById("route-markers"),
  mapEmpty: document.getElementById("map-empty"),
  metricGrid: document.getElementById("metric-grid"),
  lastUpdated: document.getElementById("last-updated"),
};

const state = { apiBase: "", trips: [], health: null };

function setConnectionStatus(kind, message) {
  elements.connectionState.dataset.state = kind;
  elements.connectionLabel.textContent = message;
}

function setHelp(message, tone = "") {
  elements.connectionHelp.textContent = message;
  if (tone) elements.connectionHelp.dataset.tone = tone;
  else delete elements.connectionHelp.dataset.tone;
}

function setRequestNote(message, tone = "") {
  elements.requestNote.textContent = message;
  if (tone) elements.requestNote.dataset.tone = tone;
  else delete elements.requestNote.dataset.tone;
}

function normalizeApiBase(value) {
  const base = value.trim().replace(/\/+$/, "");
  if (!base) throw new Error("Enter Laptop 1’s API address.");
  let parsed;
  try {
    parsed = new URL(base);
  } catch {
    throw new Error("Enter a full address, for example http://192.168.1.10:8000.");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("Use an HTTP or HTTPS address for the API.");
  }
  return base;
}

async function requestJson(path, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(`${state.apiBase}${path}`, {
      ...options,
      signal: controller.signal,
      headers: { Accept: "application/json", ...(options.headers || {}) },
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const message = payload && typeof payload.detail === "string" ? payload.detail : `API request returned ${response.status}.`;
      throw new Error(message);
    }
    return payload;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The API request timed out. Check Laptop 1 and try again.");
    if (error instanceof TypeError) {
      throw new Error("Could not reach the API. Check Laptop 1’s address, port 8000, and shared network.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function renderTrips(trips) {
  elements.tripSelect.replaceChildren();
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = trips.length ? "Choose a trip" : "No trips available";
  elements.tripSelect.append(placeholder);

  for (const trip of trips) {
    if (!trip || typeof trip.trip_id !== "string") continue;
    const option = document.createElement("option");
    option.value = trip.trip_id;
    const count = Number.isFinite(Number(trip.n_points)) ? Number(trip.n_points).toLocaleString() : "—";
    option.textContent = `Trip ${trip.trip_id} · ${count} GPS points`;
    elements.tripSelect.append(option);
  }
  elements.tripSelect.disabled = trips.length === 0;
  elements.tripCount.textContent = `${trips.length} ${trips.length === 1 ? "trip" : "trips"}`;
  elements.runButton.disabled = true;
}

async function connectToApi() {
  let base;
  try {
    base = normalizeApiBase(elements.apiBase.value);
  } catch (error) {
    setConnectionStatus("error", "API address needed");
    setHelp(error.message, "error");
    return;
  }

  state.apiBase = base;
  localStorage.setItem(API_STORAGE_KEY, base);
  elements.connectButton.disabled = true;
  elements.connectButton.textContent = "Connecting…";
  setConnectionStatus("busy", "Connecting to API");
  setHelp("Checking API health and loading the available trips.");
  setRequestNote("Waiting for a trip selection.");

  try {
    const [health, tripResponse] = await Promise.all([
      requestJson("/health"),
      requestJson("/trips"),
    ]);
    if (!health || health.status !== "ok") throw new Error("The API health check did not return an OK status.");
    const trips = Array.isArray(tripResponse?.trips) ? tripResponse.trips : [];
    state.health = health;
    state.trips = trips;
    renderTrips(trips);
    elements.systemModel.textContent = health.model || "Model ready";
    elements.systemDecoder.textContent = health.decoder || "—";
    setConnectionStatus("connected", "API connected");
    setHelp(`Connected to ${base}. ${trips.length} available ${trips.length === 1 ? "trip" : "trips"}.`);
    setRequestNote(trips.length ? "Choose a trip to begin inference." : "The API is connected, but it returned no trips.");
    elements.lastUpdated.textContent = `Connected · ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  } catch (error) {
    state.health = null;
    state.trips = [];
    renderTrips([]);
    elements.systemModel.textContent = "Waiting for API";
    elements.systemDecoder.textContent = "—";
    setConnectionStatus("error", "Connection failed");
    setHelp(error.message, "error");
    setRequestNote("Prediction is unavailable until the API connection works.", "error");
  } finally {
    elements.connectButton.disabled = false;
    elements.connectButton.textContent = "Connect";
  }
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}

function validLatLon(point) {
  if (!Array.isArray(point) || point.length < 2) return false;
  const lat = Number(point[0]);
  const lon = Number(point[1]);
  return Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180;
}

function projectRoute(gpsInput, routeInput, segmentsInput) {
  const gps = Array.isArray(gpsInput) ? gpsInput.filter(validLatLon).map(([lat, lon]) => [Number(lat), Number(lon)]) : [];
  const sourceSegments = Array.isArray(segmentsInput) ? segmentsInput : [routeInput];
  const segments = sourceSegments.filter(Array.isArray).map(segment =>
    segment.filter(validLatLon).map(([lat, lon]) => [Number(lat), Number(lon)]));
  const route = segments.flat();
  const combined = [...gps, ...route];
  if (!combined.length) return null;

  const meanLat = combined.reduce((sum, point) => sum + point[0], 0) / combined.length;
  const longitudeScale = Math.max(Math.cos(meanLat * Math.PI / 180), 0.01);
  const xy = combined.map(([lat, lon]) => [lon * longitudeScale, lat]);
  const xs = xy.map(point => point[0]);
  const ys = xy.map(point => point[1]);
  let minX = Math.min(...xs), maxX = Math.max(...xs);
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  if (maxX - minX < 1e-9) { minX -= 0.00001; maxX += 0.00001; }
  if (maxY - minY < 1e-9) { minY -= 0.00001; maxY += 0.00001; }

  const width = 1000, height = 600, margin = 58;
  const scale = Math.min((width - margin * 2) / (maxX - minX), (height - margin * 2) / (maxY - minY));
  const drawnWidth = (maxX - minX) * scale;
  const drawnHeight = (maxY - minY) * scale;
  const left = (width - drawnWidth) / 2;
  const top = (height - drawnHeight) / 2;
  const toScreen = ([lat, lon]) => ({
    x: left + (lon * longitudeScale - minX) * scale,
    y: height - (top + (lat - minY) * scale),
  });
  return { gps, route, segments, toScreen, width, height };
}

function pathData(points, toScreen) {
  if (!points.length) return "";
  return points.map((point, index) => {
    const p = toScreen(point);
    return `${index ? "L" : "M"}${p.x.toFixed(2)},${p.y.toFixed(2)}`;
  }).join(" ");
}

function drawMap(gpsInput, routeInput, segmentsInput) {
  elements.mapGrid.replaceChildren();
  elements.gpsPoints.replaceChildren();
  elements.routeMarkers.replaceChildren();
  elements.gpsTrace.setAttribute("d", "");
  elements.predictedPath.setAttribute("d", "");

  const plot = projectRoute(gpsInput, routeInput, segmentsInput);
  if (!plot) {
    elements.routeMap.classList.add("hidden");
    elements.mapEmpty.classList.remove("hidden");
    elements.mapSummary.textContent = "Route coordinates unavailable";
    return;
  }

  elements.routeMap.classList.remove("hidden");
  elements.mapEmpty.classList.add("hidden");

  for (let i = 1; i <= 5; i += 1) {
    const x = (plot.width * i) / 6;
    const y = (plot.height * i) / 6;
    elements.mapGrid.append(svgElement("line", { x1: x, y1: 35, x2: x, y2: 565, class: "grid-line" }));
    elements.mapGrid.append(svgElement("line", { x1: 48, y1: y, x2: 970, y2: y, class: "grid-line" }));
  }

  const sampledGps = plot.gps.length > 1200
    ? plot.gps.filter((_, index) => index % Math.ceil(plot.gps.length / 1200) === 0)
    : plot.gps;
  elements.gpsTrace.setAttribute("d", pathData(sampledGps, plot.toScreen));
  for (const point of sampledGps) {
    const p = plot.toScreen(point);
    elements.gpsPoints.append(svgElement("circle", { cx: p.x, cy: p.y, r: 2.15, class: "gps-point" }));
  }

  elements.predictedPath.setAttribute("d",
    plot.segments.map(segment => pathData(segment, plot.toScreen)).join(" "));
  if (plot.route.length) {
    const first = plot.toScreen(plot.route[0]);
    const last = plot.toScreen(plot.route[plot.route.length - 1]);
    elements.routeMarkers.append(svgElement("circle", { cx: first.x, cy: first.y, r: 6, class: "route-start" }));
    elements.routeMarkers.append(svgElement("circle", { cx: last.x, cy: last.y, r: 5, class: "route-end" }));
  }

  elements.mapSummary.textContent = `${plot.gps.length.toLocaleString()} GPS observations · ${plot.route.length.toLocaleString()} predicted route coordinates · ${plot.segments.length} segment(s)`;
  elements.mapCaption.textContent = "Longitude and latitude · gaps are not connected; intermediate road connectors are display-only";
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
}

function formatNumber(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function addMetric(label, value, accent = false) {
  const card = document.createElement("article");
  card.className = "metric-card";
  const name = document.createElement("span");
  name.className = "metric-label";
  name.textContent = label;
  const result = document.createElement("strong");
  result.className = `metric-value${accent ? " accent" : ""}`;
  result.textContent = value;
  card.append(name, result);
  elements.metricGrid.append(card);
}

function renderMetrics(metrics, timings) {
  elements.metricGrid.replaceChildren();
  if (!metrics || typeof metrics !== "object") {
    addMetric("Reference metrics", "Unavailable", true);
    return;
  }
  addMetric("Top-1 accuracy", formatPercent(metrics.top1_accuracy), true);
  addMetric("Classifier-only Top-1", formatPercent(metrics.top1_accuracy_classifier_only));
  addMetric("Route continuity", formatPercent(metrics.route_continuity), true);
  addMetric("Candidate miss rate", formatPercent(metrics.candidate_miss_rate));
  addMetric("Route arc precision", formatPercent(metrics.route_arc_precision));
  addMetric("Route arc recall", formatPercent(metrics.route_arc_recall));
  addMetric("Route arc F1", formatPercent(metrics.route_arc_f1), true);
  addMetric("Pipeline time", `${formatNumber(timings?.total_ms, 1)} ms`);
}

function renderPrediction(result) {
  elements.emptyState.classList.add("hidden");
  elements.resultsContent.classList.remove("hidden");
  elements.resultTripId.textContent = result.trip_id || elements.tripSelect.value || "—";
  elements.resultModel.textContent = `${result.model || "Model"} · ${result.decoder || "decoder unavailable"}`;
  drawMap(result.gps, result.predicted_route, result.predicted_route_segments);
  renderMetrics(result.metrics, result.timings_ms);
  elements.lastUpdated.textContent = `Prediction complete · ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

async function runPrediction() {
  const tripId = elements.tripSelect.value;
  if (!tripId) return;
  elements.runButton.disabled = true;
  elements.connectButton.disabled = true;
  elements.runButton.querySelector("span:first-child").textContent = "Processing trip…";
  setRequestNote(`Sending trip ${tripId} to the API.`);
  try {
    const result = await requestJson("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: tripId }),
    });
    renderPrediction(result);
    setRequestNote(`Prediction returned for trip ${tripId}.`, "success");
    setConnectionStatus("connected", "API connected");
  } catch (error) {
    setRequestNote(error.message, "error");
    setConnectionStatus("error", "Prediction failed");
  } finally {
    elements.runButton.querySelector("span:first-child").textContent = "Run prediction";
    elements.runButton.disabled = !elements.tripSelect.value;
    elements.connectButton.disabled = false;
  }
}

elements.connectForm.addEventListener("submit", event => {
  event.preventDefault();
  connectToApi();
});

elements.tripSelect.addEventListener("change", () => {
  elements.runButton.disabled = !elements.tripSelect.value;
  if (elements.tripSelect.value) setRequestNote(`Ready to predict trip ${elements.tripSelect.value}.`);
});

elements.runButton.addEventListener("click", runPrediction);

const savedApiBase = localStorage.getItem(API_STORAGE_KEY);
if (savedApiBase) elements.apiBase.value = savedApiBase;
