const $ = (sel) => document.querySelector(sel);

const state = { ws: null, duration: 0, lanes: [], truthBoxes: [], selection: null };

// end time (seconds) at which the current playback must stop, or null. We
// can't rely on ws.play(start,end)'s end: it's stored in a private
// `stopAtPosition` that the media `seeking` handler nulls out, and whether
// that seek fires before or after the stop is saved is a microtask race ->
// intermittent play-to-end. So we stop ourselves on timeupdate (rAF-driven,
// ~60 Hz) instead.
let playStopAt = null;
let ignoreSeek = 0, ignoreSeekT = null;
function armIgnoreSeek() {
  ignoreSeek = 1;
  clearTimeout(ignoreSeekT);
  ignoreSeekT = setTimeout(() => { ignoreSeek = 0; }, 500);
}
function playRegion(start, end) {
  const ws = state.ws;
  if (!ws) return;
  const dur = ws.getDuration();
  state.selection = { start, end };   // what the play button replays
  playStopAt = Math.max(0, Math.min(end - 1e-6, dur));
  armIgnoreSeek();
  ws.seekTo(Math.max(0, Math.min(start, dur)) / dur);
  ws.play().catch(() => {});
  syncPlayButton();
}
// toggle button: ▶/⏸ reflects live playback so a click does the opposite of
// what you see. While playing, a click stops; while stopped, it plays the
// recorded selection (or replays it from the start).
function syncPlayButton() {
  const btn = $("#playstop");
  if (!btn) return;
  if (!state.ws) {             // no clip loaded (between switches) -> idle glyph
    btn.textContent = "▶";
    btn.classList.remove("playing");
    btn.title = "play";
    return;
  }
  const playing = state.ws.isPlaying?.() ?? false;
  btn.textContent = playing ? "⏸" : "▶";
  btn.classList.toggle("playing", playing);
  btn.title = playing ? "stop"
    : (state.selection ? "play selection" : "play from here");
}
function onPlayStopClick() {
  const ws = state.ws;
  if (!ws) return;
  if (ws.isPlaying?.()) { ws.pause(); syncPlayButton(); return; }
  const sel = state.selection;
  const dur = ws.getDuration();
  if (sel) {
    // start (or restart) the recorded region from its beginning
    playStopAt = Math.max(0, Math.min(sel.end - 1e-6, dur));
    armIgnoreSeek();
    ws.seekTo(Math.max(0, Math.min(sel.start, dur)) / dur);
    ws.play().catch(() => {});
  } else {
    // nothing selected yet: play from wherever the cursor is, no auto-stop
    playStopAt = null;
    armIgnoreSeek();
    ws.play().catch(() => {});
  }
  syncPlayButton();
}
function enforceRegionStop(t) {
  if (playStopAt == null || !state.ws) return;
  if (t >= playStopAt) {
    const stop = playStopAt;
    playStopAt = null;
    armIgnoreSeek();       // our own setPosition seek shouldn't cancel a future stop
    state.ws.pause();
    state.ws.setTime(stop);
    syncPlayButton();      // auto-stop flips the button back to ▶
  }
}

async function loadClips() {
  const clips = await (await fetch("/api/clips")).json();
  const nav = $("#sidebar");
  clips.forEach((c) => {
    const b = document.createElement("button");
    b.className = "clip" + (c.name.endsWith("-uvr") ? " uvr" : "");
    b.textContent = c.name;
    b.title = c.media ? "media: " + c.media : "(no media file)";
    b.onclick = () => selectClip(c, b);
    nav.appendChild(b);
  });
  if (clips.length) selectClip(clips[0], nav.firstChild);
}

function wordsOf(json) {
  const out = [];
  for (const seg of json.segments ?? []) for (const w of seg.words ?? []) out.push(w);
  return out;
}

async function selectClip(clip, btn) {
  document.querySelectorAll(".clip").forEach((b) => b.classList.remove("sel"));
  btn.classList.add("sel");
  $("#clip-name").textContent = clip.name;
  $("#waveform").innerHTML = "";
  $("#lanescroll-inner").innerHTML = "";
  $("#now").textContent = "";
  $("#zoom").value = 30;
  $("#meta").textContent = clip.media ?? "no media file in clip dir";
  if (state.ws) {
    // backend: "WebAudio" hands wavesurfer an *external* WebAudioPlayer as its
    // media element, and wavesurfer's destroy() deliberately leaves external
    // media alone -- the running AudioBufferSourceNode (and the whole
    // AudioContext) would leak, letting the previous clip keep playing under
    // the new one. Stop playback and destroy the player explicitly.
    const old = state.ws;
    try { old.pause(); } catch {}
    old.destroy();
    const media = typeof old.getMediaElement === "function" ? old.getMediaElement() : null;
    if (media && typeof media.destroy === "function") media.destroy();
  }
  if (state.cleanupRegion) state.cleanupRegion();  // unbind stale wheel zoom
  playStopAt = null;
  state.ws = null;
  state.lanes = [];
  state.truthBoxes = [];
  state.selection = null;
  syncPlayButton();

  if (!clip.media) return;
  const ws = WaveSurfer.create({
    container: "#waveform",
    url: "/" + clip.name + "/" + clip.media,
    height: 100,
    minPxPerSec: 30,
    // The hidden <audio> element (the default backend) draws the waveform from
    // a sample-accurate decodeAudioData but PLAYBACK/SPEAKING rides on the
    // media element, whose seek clock drifts from the decoded sample grid for
    // lossy codes (MP3/OGG) -- by encoder-delay + accumulated frame offset
    // (measured: arnold.mp3 ~0.08s late, the long VBR rye.mp3 up to ~1.3s late
    // depending on position; PCM/byte-offset wav stays exact). That's why the
    // highlighted slice looked right but the audio wasn't on it. Forcing the
    // WebAudio backend plays an AudioBufferSourceNode from the same decode the
    // waveform is drawn from, so audio == picture for every codec.
    backend: "WebAudio",
    waveColor: "#3a4150",
    progressColor: "#7aa2f7",
    cursorColor: "#f7768e",
  });
  state.ws = ws;
  ws.on("ready", (dur) => {
    state.duration = dur;
    placeBoxes();   // re-place with the confirmed duration (may have
                     // been placed earlier with a provisional one)
    hookScroller(ws);
    syncLayout();
  });
  ws.on("redrawcomplete", syncLayout);  // after canvas reaches final width
  ws.on("timeupdate", (t) => { setCursor(t); updateTime(t); enforceRegionStop(t); });
  ws.on("play", () => { syncPlayButton(); });   // any playback -> show stop
  ws.on("pause", () => { playStopAt = null; syncPlayButton(); });   // user clicked pause region stop
  ws.on("seeking", () => { if (!ignoreSeek) playStopAt = null; });  // user seek releases the stop
  ws.on("finish", () => { playStopAt = null; setCursor(state.duration); updateTime(state.duration); syncPlayButton(); });
  state.cleanupRegion = wireRegionDrag(ws);

  try {
    const truth = await (await fetch(`/${clip.name}/truth.json`)).json();
    addLane("truth", wordsOf(truth), truth, true);
    for (const cand of clip.candidates) {
      const r = await fetch(`/${clip.name}/${cand}`);
      if (!r.ok) continue;
      addLane(cand, wordsOf(await r.json()), cand);
    }
    placeBoxes();
  } catch (e) {
    $("#meta").textContent = "load error: " + e;
  }
}

function hookScroller(ws) {
  const scroller = ws.getWrapper()?.parentElement;
  if (!scroller || scroller.dataset.hooked) return;
  scroller.dataset.hooked = "1";
  scroller.addEventListener("scroll", () => {
    $("#lanescroll").scrollLeft = scroller.scrollLeft;
  }, { passive: true });
}

// drag across the waveform to select & play a region. The library's
// dragstart/drag/dragend only fire when the (off-by-default) `dragToSeek`
// option is set, so we do our own pointer tracking on the waveform element:
// a press+move past a threshold is a region; a press+release without moving
// is left untouched (the library's own click handler seeks to it). The played
// region is drawn as a band across all lanes (light DOM, same %-coords as the
// word boxes) so the highlighted words line up exactly with it.
const DRAG_MIN_PX = 6;
function wireRegionDrag(ws) {
  let drag = null, downX = 0, suppressNextClick = false;
  const sel = document.createElement("div");
  sel.className = "play-sel";
  sel.style.display = "none";
  $("#lanescroll-inner").appendChild(sel);

  const toNorm = (clientX) => {
    const w = ws.getWrapper().getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - w.left) / w.width));
  };
  const showSel = (a, b) => {
    const lo = Math.min(a, b), hi = Math.max(a, b);
    sel.style.display = "block";
    sel.style.left = (lo * 100) + "%";
    sel.style.width = ((hi - lo) * 100) + "%";
  };

  const onDown = (e) => {
    if (e.button !== 0) return;
    if (e.target && e.target.closest && e.target.closest(".word")) return;
    drag = { start: toNorm(e.clientX), end: toNorm(e.clientX), moved: false };
    downX = e.clientX;
  };
  const onMove = (e) => {
    if (!drag) return;
    if (!drag.moved && Math.abs(e.clientX - downX) < DRAG_MIN_PX) return;
    drag.moved = true;
    drag.end = toNorm(e.clientX);
    showSel(drag.start, drag.end);
  };
  const onUp = (e) => {
    if (!drag) return;
    const { start, end, moved } = drag;
    drag = null;
    if (!moved) return;             // plain click -> library seeks
    suppressNextClick = true;       // block the library's trailing click-seek
    showSel(start, end);
    const dur = state.duration;     // drag coords are normalized; playRegion wants seconds
    playRegion(start * dur, end * dur);
  };
  // a real drag also fires a native `click` on the wrapper, which the library
  // turns into seekTo(releaseX) -- that would clobber the region we just
  // started playing. A capturing listener on the wrapper runs before the
  // library's bubbling click handler, so swallow it when a drag just ended.
  const wrap = ws.getWrapper();
  const onCaptureClick = (e) => {
    if (suppressNextClick) { e.stopPropagation(); suppressNextClick = false; }
  };
  if (wrap) wrap.addEventListener("click", onCaptureClick, true);

  const wf = $("#waveform");
  wf.addEventListener("pointerdown", onDown);
  window.addEventListener("pointermove", onMove, { passive: true });
  window.addEventListener("pointerup", onUp);

  // wheel over the waveform = zoom, anchored at the cursor (v7 core has no
  // built-in wheel zoom; the slider is the coarse equivalent). scrollWidth is
  // async after ws.zoom (canvas redraw), so compute the target width
  // analytically and apply scrollLeft in a one-shot `redrawcomplete` callback.
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const scrollerEl = () => ws.getWrapper()?.parentElement;
  const onWheel = (e) => {
    const sc = scrollerEl();
    const dur = ws.getDuration();
    if (!sc || !dur) return;
    const ppsBefore = sc.scrollWidth / dur;         // px/sec (already rendered)
    const newPps = clamp(ppsBefore * (e.deltaY < 0 ? 1.25 : 0.8), 30, 2000);
    if (Math.abs(newPps - ppsBefore) < 1e-6) return;
    const rect = sc.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const t = (sc.scrollLeft + mouseX) / ppsBefore; // seconds under cursor
    e.preventDefault();
    ws.zoom(newPps);
    $("#zoom").value = Math.round(newPps);
    const newW = Math.max(sc.clientWidth, dur * newPps);
    const target = clamp(t * (newW / dur) - mouseX, 0, Math.max(0, newW - sc.clientWidth));
    ws.once("redrawcomplete", () => { sc.scrollLeft = target; syncLayout(); });
  };
  $("#waveform").addEventListener("wheel", onWheel, { passive: false });

  // #waveform / window / wrapper survive clip switches, so hand back a
  // teardown for when this clip is destroyed (selectClip calls state.cleanupRegion)
  return () => {
    wf.removeEventListener("wheel", onWheel);
    wf.removeEventListener("pointerdown", onDown);
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    if (wrap) wrap.removeEventListener("click", onCaptureClick, true);
  };
}

// make the lane area as wide as the (zoomable) waveform scroller so the
// %-positioned lanes map to the same pixels as the waveform
function syncLayout() {
  const scroller = state.ws?.getWrapper()?.parentElement;
  const w = scroller?.scrollWidth ?? 0;
  if (w > 0) $("#lanescroll-inner").style.width = w + "px";
}

function addLane(name, words, json, isTruth = false) {
  const display = name.replace(/\.json$/, ""); // all are .json; drop the suffix
  const row = document.createElement("div");
  row.className = "lane" + (isTruth ? " truthlane" : "");
  const label = document.createElement("div");
  label.className = "lane-label";
  label.textContent = display;
  if (!isTruth && json.timeline_quality) {
    label.title = json.timeline_quality +
      (json.timeline_degraded_reason ? " — " + json.timeline_degraded_reason : "");
    if (!json.timeline_quality.startsWith("forced")) label.classList.add("native");
  }
  const body = document.createElement("div");
  body.className = "lane-body";
  row.append(label, body);
  $("#lanescroll-inner").appendChild(row);

  const boxes = [];
  for (const w of words) {
    const el = document.createElement("div");
    el.className = "word";
    el.textContent = w.word.trim() || "·";
    el.title = `${w.word}  [${w.start.toFixed(3)} – ${w.end.toFixed(3)}]`;
    if (w.confidence !== undefined) el.dataset.conf = w.confidence;
    el.onclick = (e) => {
      e.stopPropagation();
      if (state.duration) state.ws.seekTo(w.start / state.duration);
    };
    el.ondblclick = (e) => {
      e.stopPropagation();
      if (state.duration) playRegion(w.start, w.end);  // robust region stop
    };
    body.appendChild(el);
    boxes.push({ el, start: w.start, end: w.end, text: w.word });
  }
  const cursor = document.createElement("div");
  cursor.className = "cursor";
  body.appendChild(cursor);
  const lane = { name: display, cursor, boxes, truth: isTruth };
  state.lanes.push(lane);

  const markMisses = (truth, l) => l.boxes.forEach((b) => {
    b.el.classList.toggle("miss", !truth.some((t) => b.start < t.end && b.end > t.start));
  });
  if (isTruth) {
    state.truthBoxes = boxes;
  } else if (state.truthBoxes.length) {
    markMisses(state.truthBoxes, lane);
  }
}

// position every box at (start/dur)·100% of the full lane width; the wave
// scroller and the lane area share that width, so x matches the waveform.
function placeBoxes() {
  const dur = state.duration;
  if (!dur) return;
  state.placedDur = dur;
  for (const lane of state.lanes) {
    for (const b of lane.boxes) {
      b.el.style.left = (b.start / dur) * 100 + "%";
      b.el.style.width = Math.max(0.01, (b.end - b.start) / dur * 100) + "%";
    }
  }
}

function setCursor(t) {
  if (!state.duration) return;
  const pct = (t / state.duration) * 100;
  for (const lane of state.lanes) {
    lane.cursor.style.left = pct + "%";
    let cur = null;
    for (const b of lane.boxes) {
      if (t < b.start) break;
      if (t <= b.end) cur = b;
    }
    for (const b of lane.boxes) b.el.classList.toggle("active", b === cur);
  }
}

function updateTime(t) {
  const lines = [`t = ${t.toFixed(3)}`];
  for (const lane of state.lanes) {
    const cur = lane.boxes.find((b) => t >= b.start && t <= b.end);
    if (cur) lines.push(`${lane.name}: "${cur.text}" [${cur.start.toFixed(2)}–${cur.end.toFixed(2)}]`);
  }
  $("#now").textContent = lines.join("\n");
}

$("#lanescroll").addEventListener("scroll", () => {
  const scroller = state.ws?.getWrapper()?.parentElement;
  if (scroller) scroller.scrollLeft = $("#lanescroll").scrollLeft;
}, { passive: true });

$("#zoom").addEventListener("input", (e) => {
  state.ws?.zoom(+e.target.value);   // px/sec; wavesurfer clamps to minPxPerSec
});

$("#playstop").addEventListener("click", onPlayStopClick);

loadClips();
