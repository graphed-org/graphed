"use strict";
// graphed live-execution dashboard SPA. Python emits JSON over SSE; we render with uPlot + d3-flame-graph.
// Frame types: "snapshot" (full state on connect), "stat" (counters/throughput point), "flame" (profile tree).

const $ = (id) => document.getElementById(id);

// --- time-series state (accumulated client-side from stat frames) ---
const xs = [];           // elapsed seconds
const finished = [];     // cumulative tasks finished
const inflight = [];     // tasks in flight

function mkChart(el, label, stroke) {
  const opts = {
    width: el.clientWidth || 460, height: 200,
    cursor: { drag: { x: true, y: false } },
    scales: { x: { time: false } },
    axes: [
      { stroke: "#8b93a5", grid: { stroke: "#262b36" }, ticks: { stroke: "#262b36" } },
      { stroke: "#8b93a5", grid: { stroke: "#262b36" }, ticks: { stroke: "#262b36" } },
    ],
    series: [
      { label: "t (s)" },
      { label, stroke, width: 2, fill: stroke + "22" },
    ],
  };
  return new uPlot(opts, [[0], [0]], el);
}

const thru = mkChart($("thru"), "finished", "#5b9dff");
const fly = mkChart($("fly"), "in flight", "#3fb950");

function redraw() {
  thru.setData([xs, finished]);
  fly.setData([xs, inflight]);
}

// --- flamegraph ---
let flameChart = null;
function renderFlame(tree) {
  if (!tree || !tree.children || tree.children.length === 0) {
    $("flamehint").textContent = "(no samples yet — run with profile=True)";
    return;
  }
  $("flamehint").textContent = "";
  const el = $("flame");
  if (flameChart) { el.innerHTML = ""; }
  flameChart = flamegraph().width(el.clientWidth || 900).cellHeight(18).minFrameSize(1);
  d3.select(el).datum(tree).call(flameChart);
}

// --- stat/counters ---
function applyStat(s) {
  $("done").textContent = s.counters.finished;
  $("sub").textContent = s.counters.submitted;
  $("inflight").textContent = s.inflight;
  $("errored").textContent = s.counters.errored;
  $("combines").textContent = s.counters.combines;
  $("elapsed").textContent = s.elapsed.toFixed(1);
  const denom = s.counters.submitted || 1;
  $("bar").style.width = Math.min(100, (100 * s.counters.finished) / denom) + "%";
  // worker table
  const rows = Object.keys(s.workers).sort().map((w) => {
    const v = s.workers[w];
    return `<tr><td>${w}</td><td>${v.started}</td><td>${v.finished}</td><td>${v.errored}</td><td>${v.entries}</td></tr>`;
  });
  $("workers").innerHTML = rows.join("");
  if (s.last_error) {
    $("err").classList.remove("muted");
    $("err").textContent = `task ${s.last_error.key} @ ${s.last_error.worker}\n${s.last_error.message}`;
  }
}

function pushPoint(elapsed, fin, fly_) {
  xs.push(elapsed); finished.push(fin); inflight.push(fly_);
}

// --- SSE wiring ---
function connect() {
  const es = new EventSource("/events");
  es.onopen = () => { $("conn").textContent = "● live"; $("conn").style.color = "#3fb950"; };
  es.onerror = () => { $("conn").textContent = "● reconnecting…"; $("conn").style.color = "#f85149"; };
  es.onmessage = (ev) => {
    const f = JSON.parse(ev.data);
    if (f.type === "snapshot") {
      xs.length = finished.length = inflight.length = 0;
      (f.throughput || []).forEach(([t, n]) => pushPoint(t, n, 0));
      applyStat(f);
      renderFlame(f.flame);
      redraw();
    } else if (f.type === "stat") {
      if (f.throughput) pushPoint(f.throughput[0], f.throughput[1], f.inflight);
      else if (xs.length) pushPoint(f.elapsed, finished[finished.length - 1], f.inflight);
      applyStat(f);
      redraw();
    } else if (f.type === "flame") {
      renderFlame(f.flame);
    }
  };
}
connect();
window.addEventListener("resize", () => { thru.setSize({ width: $("thru").clientWidth, height: 200 });
  fly.setSize({ width: $("fly").clientWidth, height: 200 }); });
