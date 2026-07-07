const params = new URLSearchParams(location.search);
const WS_URL = params.get("ws") || `ws://${location.hostname || "localhost"}:8000/ws/solve`;
const HTTP_BASE = params.get("http") || `http://${location.hostname || "localhost"}:8000`;
const MAX_STEPS = Number(params.get("max_steps") || 100);
const FALLBACK_CANVAS_PX = 320; // only used if the wrapper hasn't been laid out yet
const MIN_CELL_PX = 16;
const MOVE_DURATION_MS = 320;
const MAX_MOVE_DURATION_MS = 900; // cap animation length for long multi-cell moves
const SENSE_PAUSE_MS = 350;
const FLASH_DURATION_MS = 900;

const DIRECTION_DELTAS = {
  up: [0, -1],
  down: [0, 1],
  left: [-1, 0],
  right: [1, 0],
};

const canvas = document.getElementById("maze-canvas");
const canvasWrapper = document.getElementById("canvas-wrapper");
const ctx = canvas.getContext("2d");
const logEntries = document.getElementById("log-entries");
const statusText = document.getElementById("status-text");
const stepCount = document.getElementById("step-count");
const rowsInput = document.getElementById("rows-input");
const colsInput = document.getElementById("cols-input");
const generateBtn = document.getElementById("generate-btn");
const solveBtn = document.getElementById("solve-btn");
const stopBtn = document.getElementById("stop-btn");
const checkConnBtn = document.getElementById("check-conn-btn");
const solverGroup = document.getElementById("solver-group");
const solverButtons = solverGroup ? Array.from(solverGroup.querySelectorAll(".solver-btn")) : [];
let selectedSolver = "llm";
const llmProviderEl = document.getElementById("llm-provider");
const llmBaseUrlEl = document.getElementById("llm-base-url");
const llmModelEl = document.getElementById("llm-model");
const llmCheckResultEl = document.getElementById("llm-check-result");
const memoryContentEl = document.getElementById("memory-content");

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const COLORS = {
  unknown: cssVar("--color-unknown"),
  open: cssVar("--color-open"),
  wall: cssVar("--color-wall"),
  wallHatch: cssVar("--color-wall-hatch"),
  start: cssVar("--color-start"),
  trail: cssVar("--color-trail"),
  gridLine: cssVar("--color-grid-line"),
};

const board = {
  width: 0,
  height: 0,
  cellPx: 32,
  start: [0, 0],
  end: [0, 0],
  walls: new Set(), // "x,y" ground-truth walls, always rendered — the robot still only learns them via sensing
  exploredOpen: [], // 2D bool: cell confirmed open by sense_surroundings
  sensedWallHatch: [], // 2D bool: wall cell specifically confirmed by sense_surroundings
  robotPos: { x: 0, y: 0 },
  tween: null, // {fromX, fromY, toX, toY, startTime, duration, resolve}
  flashes: [], // {x, y, startTime}
  trail: [], // [{x, y}, ...] waypoints the robot has actually reached, in order
};

let currentMaze = null; // last layout returned by /maze/generate, sent verbatim to /ws/solve
let socket = null;
let solving = false;
let finished = false;
let checkingConnection = false;
const eventQueue = [];
let pumping = false;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function wallKey(x, y) {
  return `${x},${y}`;
}

function isWall(x, y) {
  return board.walls.has(wallKey(x, y));
}

function computeCellPx(mazeWidth, mazeHeight) {
  // Fit both dimensions inside the wrapper's actual rendered box, keeping
  // square cells (same cellPx on both axes) so the maze's real aspect ratio
  // is preserved regardless of how many rows/columns it has.
  const availableWidth = canvasWrapper.clientWidth || FALLBACK_CANVAS_PX;
  const availableHeight = canvasWrapper.clientHeight || FALLBACK_CANVAS_PX;
  const fitted = Math.floor(Math.min(availableWidth / mazeWidth, availableHeight / mazeHeight));
  return Math.max(MIN_CELL_PX, fitted);
}

function resizeCanvasToFit() {
  if (!board.width || !board.height) return;
  board.cellPx = computeCellPx(board.width, board.height);
  canvas.width = board.width * board.cellPx;
  canvas.height = board.height * board.cellPx;
}

function applyMazeLayout(data) {
  board.width = data.width;
  board.height = data.height;
  board.start = data.start;
  board.end = data.end;
  board.walls = new Set(data.walls.map(([x, y]) => wallKey(x, y)));
  board.exploredOpen = Array.from({ length: data.height }, () => Array(data.width).fill(false));
  board.sensedWallHatch = Array.from({ length: data.height }, () => Array(data.width).fill(false));
  board.robotPos = { x: data.start[0], y: data.start[1] };
  board.tween = null;
  board.flashes = [];
  board.trail = [{ x: data.start[0], y: data.start[1] }];
  resizeCanvasToFit();
}

function revealCell(x, y, kind) {
  if (x < 0 || y < 0 || x >= board.width || y >= board.height) return;
  if (kind === "wall") {
    board.sensedWallHatch[y][x] = true;
  } else if (kind === "open" || kind === "end") {
    board.exploredOpen[y][x] = true;
  }
  board.flashes.push({ x, y, startTime: performance.now() });
}

function setStatus(cls, text) {
  statusText.className = cls;
  statusText.textContent = text;
}

function setControlsState() {
  generateBtn.disabled = solving;
  rowsInput.disabled = solving;
  colsInput.disabled = solving;
  solveBtn.disabled = solving || !currentMaze;
  stopBtn.disabled = !solving;
  checkConnBtn.disabled = solving || checkingConnection;
  solverButtons.forEach((btn) => (btn.disabled = solving));
}

function selectSolver(solver) {
  selectedSolver = solver;
  solverButtons.forEach((btn) => {
    const active = btn.dataset.solver === solver;
    btn.classList.toggle("selected", active);
    btn.setAttribute("aria-pressed", String(active));
  });
}

function clearLog() {
  logEntries.innerHTML = "";
}

function tweenRobotTo(toX, toY, duration) {
  return new Promise((resolve) => {
    board.tween = {
      fromX: board.robotPos.x,
      fromY: board.robotPos.y,
      toX,
      toY,
      startTime: performance.now(),
      duration,
      resolve,
    };
  });
}

function easeInOutQuad(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function tick(now) {
  if (board.tween) {
    const { fromX, fromY, toX, toY, startTime, duration, resolve } = board.tween;
    const t = Math.min(1, (now - startTime) / duration);
    const eased = easeInOutQuad(t);
    board.robotPos = { x: fromX + (toX - fromX) * eased, y: fromY + (toY - fromY) * eased };
    if (t >= 1) {
      board.tween = null;
      resolve();
    }
  }
  board.flashes = board.flashes.filter((f) => now - f.startTime < FLASH_DURATION_MS);

  draw(now);
  requestAnimationFrame(tick);
}

function draw(now) {
  if (!board.width) return;
  const size = board.cellPx;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < board.height; y++) {
    for (let x = 0; x < board.width; x++) {
      const wall = isWall(x, y);
      ctx.fillStyle = wall
        ? COLORS.wall
        : isStart(x, y)
        ? COLORS.start
        : board.exploredOpen[y][x]
        ? COLORS.open
        : COLORS.unknown;
      ctx.fillRect(x * size, y * size, size, size);
      if (wall && board.sensedWallHatch[y][x]) {
        drawHatch(x, y, size);
      }
      ctx.strokeStyle = COLORS.gridLine;
      ctx.strokeRect(x * size + 0.5, y * size + 0.5, size - 1, size - 1);
    }
  }

  for (const flash of board.flashes) {
    const alpha = 1 - (now - flash.startTime) / FLASH_DURATION_MS;
    ctx.fillStyle = `rgba(96, 165, 250, ${Math.max(0, alpha * 0.35)})`;
    ctx.fillRect(flash.x * size, flash.y * size, size, size);
  }

  drawTrail(size);
  drawEndFlag(size);
  drawRobot(size);
}

function isStart(x, y) {
  return x === board.start[0] && y === board.start[1];
}

function drawHatch(x, y, size) {
  const x0 = x * size;
  const y0 = y * size;
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, size, size);
  ctx.clip();
  ctx.strokeStyle = COLORS.wallHatch;
  ctx.lineWidth = 2;
  const step = 6;
  for (let i = -size; i < size * 2; i += step) {
    ctx.beginPath();
    ctx.moveTo(x0 + i, y0);
    ctx.lineTo(x0 + i + size, y0 + size);
    ctx.stroke();
  }
  ctx.restore();
}

function cellCenter(x, y, size) {
  return [x * size + size / 2, y * size + size / 2];
}

function drawTrail(size) {
  if (!board.trail.length) return;

  ctx.save();
  ctx.strokeStyle = COLORS.trail;
  ctx.lineWidth = Math.max(2, size * 0.08);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.setLineDash([size * 0.28, size * 0.2]);
  ctx.beginPath();
  const [startX, startY] = cellCenter(board.trail[0].x, board.trail[0].y, size);
  ctx.moveTo(startX, startY);
  for (const point of board.trail.slice(1)) {
    const [px, py] = cellCenter(point.x, point.y, size);
    ctx.lineTo(px, py);
  }
  // live segment out to the robot's current (possibly mid-glide) position,
  // so the trail visibly grows in step with the move animation
  const [robotX, robotY] = cellCenter(board.robotPos.x, board.robotPos.y, size);
  ctx.lineTo(robotX, robotY);
  ctx.stroke();
  ctx.restore();

  ctx.fillStyle = COLORS.trail;
  for (const point of board.trail) {
    const [cx, cy] = cellCenter(point.x, point.y, size);
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(1.5, size * 0.05), 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawEmoji(cx, cy, emoji, size, scale) {
  ctx.font = `${size * scale}px "Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(emoji, cx, cy + size * 0.03); // slight baseline nudge — emoji glyphs sit a touch high when centered
}

function drawEndFlag(size) {
  const [x, y] = board.end;
  drawEmoji(x * size + size / 2, y * size + size / 2, "🏁", size, 0.62);
}

function drawRobot(size) {
  const cx = board.robotPos.x * size + size / 2;
  const cy = board.robotPos.y * size + size / 2;
  drawEmoji(cx, cy, "🤖", size, 0.68);
}

function logMessage(html, cls) {
  const li = document.createElement("li");
  li.className = cls;
  li.innerHTML = html;
  logEntries.appendChild(li);
  logEntries.parentElement.scrollTop = logEntries.parentElement.scrollHeight;
}

function reasoningHtml(reasoning) {
  return reasoning ? `<span class="reasoning">${escapeHtml(reasoning)}</span>` : "";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function playSense(event) {
  const [x, y] = event.position;
  const parts = [];
  for (const [dir, info] of Object.entries(event.sensed)) {
    const [dx, dy] = DIRECTION_DELTAS[dir];
    for (let i = 1; i <= info.distance; i++) {
      const kind = info.end_at === i ? "end" : "open";
      revealCell(x + dx * i, y + dy * i, kind);
    }
    if (info.blocked_by === "wall") {
      revealCell(x + dx * (info.distance + 1), y + dy * (info.distance + 1), "wall");
    }
    const endNote = info.end_at ? `, end at ${info.end_at}` : "";
    parts.push(`${dir}=${info.distance} (${info.blocked_by}${endNote})`);
  }
  logMessage(`<strong>Sense</strong> @ (${x},${y}): ${parts.join(", ")}${reasoningHtml(event.reasoning)}`, "sense");
  await sleep(SENSE_PAUSE_MS);
}

async function playMove(event) {
  const [fx, fy] = event.position_before;
  const [tx, ty] = event.position_after;
  const moved = event.distance_moved;
  if (moved > 0) {
    const [dx, dy] = DIRECTION_DELTAS[event.direction];
    for (let i = 1; i <= moved; i++) {
      revealCell(fx + dx * i, fy + dy * i, "open");
    }
    const label =
      moved === event.distance_requested
        ? `${event.direction} ×${moved}`
        : `${event.direction} ×${moved} (requested ${event.distance_requested}, blocked)`;
    logMessage(`<strong>Move</strong> ${label} → (${tx},${ty})${reasoningHtml(event.reasoning)}`, "move-ok");
    await tweenRobotTo(tx, ty, Math.min(MOVE_DURATION_MS * moved, MAX_MOVE_DURATION_MS));
    board.trail.push({ x: tx, y: ty });
  } else {
    logMessage(`<strong>Move</strong> ${event.direction} — blocked${reasoningHtml(event.reasoning)}`, "move-blocked");
    // small shake toward the blocked direction and back, for feedback
    const [dx, dy] = DIRECTION_DELTAS[event.direction];
    await tweenRobotTo(fx + dx * 0.25, fy + dy * 0.25, 120);
    await tweenRobotTo(fx, fy, 120);
  }
}

function finishSolving() {
  finished = true;
  solving = false;
  setControlsState();
  // Don't wait on the server to close its end (auto-close on coroutine return
  // is implicit and can lag) — close from here so Solve/Stop re-enable promptly.
  if (socket) socket.close();
}

function playDone(event) {
  if (event.success) {
    setStatus("success", "Reached the end!");
    logMessage(`<strong>Done</strong> — reached the end in ${event.steps} steps.`, "done-success");
  } else {
    setStatus("failed", "Did not reach the end");
    logMessage(`<strong>Done</strong> — stopped after ${event.steps} steps without reaching the end.`, "done-failed");
  }
  stepCount.textContent = `${event.steps} steps`;
  finishSolving();
}

function playError(event) {
  setStatus("failed", "Error");
  logMessage(`<strong>Error</strong>: ${escapeHtml(event.message)}`, "error");
  finishSolving();
}

function playMemory(event) {
  memoryContentEl.textContent = event.content;
  memoryContentEl.classList.remove("updated");
  void memoryContentEl.offsetWidth; // restart the CSS transition on repeated updates
  memoryContentEl.classList.add("updated");
}

async function playEvent(event) {
  switch (event.type) {
    case "memory":
      return playMemory(event);
    case "sense":
      return playSense(event);
    case "move":
      return playMove(event);
    case "done":
      return playDone(event);
    case "error":
      return playError(event);
    default:
      return undefined;
  }
}

async function pump() {
  if (pumping) return;
  pumping = true;
  while (eventQueue.length) {
    const event = eventQueue.shift();
    await playEvent(event);
  }
  pumping = false;
}

async function generateMaze() {
  const width = clamp(parseInt(colsInput.value, 10) || 4, 2, 30);
  const height = clamp(parseInt(rowsInput.value, 10) || 4, 2, 30);
  colsInput.value = width;
  rowsInput.value = height;

  setStatus("idle", "Generating maze…");
  try {
    const res = await fetch(`${HTTP_BASE}/maze/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ width, height }),
    });
    if (!res.ok) throw new Error(`generate failed: HTTP ${res.status}`);
    currentMaze = await res.json();
    applyMazeLayout(currentMaze);
    clearLog();
    stepCount.textContent = "";
    memoryContentEl.textContent = "(not started)";
    memoryContentEl.classList.remove("updated");
    setStatus("idle", "Ready to solve");
    logMessage(`Generated ${width}×${height} maze — start (${board.start.join(",")}), end (${board.end.join(",")})`, "system");
  } catch (err) {
    setStatus("failed", "Generate failed");
    logMessage(`<strong>Error</strong>: ${escapeHtml(String(err))} — is the backend running on port 8000?`, "error");
  }
  setControlsState();
}

function connectAndSolve(maze) {
  finished = false;
  eventQueue.length = 0;
  socket = new WebSocket(WS_URL);

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ maze, max_steps: MAX_STEPS, solver: selectedSolver }));
    const SOLVER_LABELS = { llm: "LLM", dstar_lite: "D* Lite" };
    const solverLabel = SOLVER_LABELS[selectedSolver] || selectedSolver.toUpperCase();
    logMessage(`Connected — solving with ${solverLabel}…`, "system");
  });

  socket.addEventListener("message", (evt) => {
    const data = JSON.parse(evt.data);
    if (data.type === "init") return; // board already applied when Solve was clicked
    eventQueue.push(data);
    pump();
  });

  socket.addEventListener("close", () => {
    solving = false;
    setControlsState();
    if (!finished) setStatus("idle", "Stopped");
    logMessage("Connection closed.", "system");
  });

  socket.addEventListener("error", () => {
    finished = true;
    setStatus("failed", "Connection error");
    logMessage("WebSocket error — is the backend running on port 8000?", "error");
  });
}

function handleSolveClick() {
  if (!currentMaze || solving) return;
  solving = true;
  setControlsState();
  clearLog();
  memoryContentEl.textContent = "Waiting for first turn…";
  memoryContentEl.classList.remove("updated");
  applyMazeLayout(currentMaze); // fresh fog-of-war + robot position for this run
  setStatus("solving", "Solving…");
  connectAndSolve(currentMaze);
}

function handleStopClick() {
  if (socket) socket.close();
}

async function loadLlmInfo() {
  try {
    const res = await fetch(`${HTTP_BASE}/llm/info`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const info = await res.json();
    llmProviderEl.textContent = info.provider;
    llmBaseUrlEl.textContent = info.base_url;
    llmModelEl.textContent = info.model;
  } catch (err) {
    llmProviderEl.textContent = "unknown";
    llmBaseUrlEl.textContent = "unknown";
    llmModelEl.textContent = "unknown";
  }
}

async function handleCheckConnectionClick() {
  checkingConnection = true;
  setControlsState();
  llmCheckResultEl.className = "checking";
  llmCheckResultEl.textContent = "Checking…";
  try {
    const res = await fetch(`${HTTP_BASE}/llm/check`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const result = await res.json();
    if (result.ok) {
      llmCheckResultEl.className = "ok";
      llmCheckResultEl.textContent = `✓ Connected — "${result.message}"`;
      logMessage(`<strong>Check connection</strong> — OK: ${escapeHtml(result.message)}`, "system");
    } else {
      llmCheckResultEl.className = "fail";
      llmCheckResultEl.textContent = `✗ ${result.message}`;
      logMessage(`<strong>Check connection</strong> — failed: ${escapeHtml(result.message)}`, "error");
    }
  } catch (err) {
    llmCheckResultEl.className = "fail";
    llmCheckResultEl.textContent = `✗ ${String(err)}`;
    logMessage(`<strong>Check connection</strong> — failed: ${escapeHtml(String(err))}`, "error");
  }
  checkingConnection = false;
  setControlsState();
}

generateBtn.addEventListener("click", generateMaze);
solveBtn.addEventListener("click", handleSolveClick);
stopBtn.addEventListener("click", handleStopClick);
checkConnBtn.addEventListener("click", handleCheckConnectionClick);
solverButtons.forEach((btn) =>
  btn.addEventListener("click", () => {
    if (!solving) selectSolver(btn.dataset.solver);
  })
);
window.addEventListener("resize", resizeCanvasToFit);

requestAnimationFrame(tick);
setControlsState();
generateMaze();
loadLlmInfo();
