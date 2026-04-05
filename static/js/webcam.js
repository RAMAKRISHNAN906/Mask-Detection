/* ═══════════════════════════════════════════════════════════════
   MaskGuard AI - Webcam Detection Module (lag-free overlay mode)
   ═══════════════════════════════════════════════════════════════ */

let webcamStream   = null;
let isDetecting    = false;
let detectionTimer = null;
let lastResults    = [];   // cached detections drawn every rAF

const video       = document.getElementById('webcam-video');
const captureCanvas = document.getElementById('webcam-canvas');   // hidden, for grabbing frames
const overlay     = document.getElementById('overlay-canvas');    // shown on top of video
const placeholder = document.getElementById('video-placeholder');
const scanOverlay = document.getElementById('scan-overlay');

// ── Overlay render loop (requestAnimationFrame — silky smooth) ─
function renderLoop() {
    if (!isDetecting) return;

    // Sync overlay size to video display size
    if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
        overlay.width  = video.videoWidth  || 640;
        overlay.height = video.videoHeight || 480;
    }

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    drawBoxes(ctx, lastResults, overlay.width, overlay.height);

    requestAnimationFrame(renderLoop);
}

// ── Detection loop — runs every 350 ms (sends small frame to server) ─
async function detectLoop() {
    if (!isDetecting) return;

    try {
        // Capture a small frame for the server (320px wide for speed)
        const vw = video.videoWidth  || 640;
        const vh = video.videoHeight || 480;
        const scale = Math.min(1, 320 / vw);
        captureCanvas.width  = Math.round(vw * scale);
        captureCanvas.height = Math.round(vh * scale);

        const ctx = captureCanvas.getContext('2d');
        ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        const imageData = captureCanvas.toDataURL('image/jpeg', 0.5);  // low quality = fast

        const t0 = performance.now();
        const response = await fetch('/api/detect/coords', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) throw new Error('Detection failed');
        const data = await response.json();
        const fps  = Math.round(1000 / (performance.now() - t0));

        if (!data.error) {
            // Scale bounding boxes from detection frame back to display frame
            const scaleBack = vw / captureCanvas.width;
            lastResults = (data.detections || []).map(d => ({
                ...d,
                box: d.box.map(v => Math.round(v * scaleBack))
            }));

            updateStats(data, fps);
            updateStatusBanner(data);
            addLogEntry(data);
            checkVoiceAlert(data);
        }
    } catch (err) {
        console.warn('Detection error:', err);
    }

    if (isDetecting) {
        detectionTimer = setTimeout(detectLoop, 350);  // detect every 350ms
    }
}

// ── Draw boxes on overlay canvas ──────────────────────────────
function drawBoxes(ctx, results, cw, ch) {
    if (!results || results.length === 0) return;

    results.forEach(d => {
        const [x1, y1, x2, y2] = d.box;
        const hasMask  = d.has_mask;
        const color    = hasMask ? '#00d26a' : '#ff3b3b';
        const label    = `${d.label} ${d.confidence}%`;

        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth   = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Corner accents
        const cLen = Math.max(14, (x2 - x1) / 6);
        ctx.lineWidth = 4;
        [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]].forEach(([cx, cy, dx, dy]) => {
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + dx * cLen, cy); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy + dy * cLen); ctx.stroke();
        });

        // Label background
        ctx.font = 'bold 14px Inter, sans-serif';
        const tw = ctx.measureText(label).width;
        const pad = 6, lh = 20;
        const lx = x1, ly = Math.max(0, y1 - lh - pad * 2);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(lx, ly, tw + pad * 2, lh + pad * 2, 6);
        ctx.fill();

        // Label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, lx + pad, ly + lh + pad - 4);
    });

    // Bottom banner bar
    const anyNoMask = results.some(d => !d.has_mask);
    const bannerColor = anyNoMask ? 'rgba(200,0,0,0.82)' : 'rgba(0,160,70,0.82)';
    const bannerText  = anyNoMask ? '✗  NO MASK FOUND' : '✓  MASK FOUND';

    const barH = Math.max(44, ch / 9);
    ctx.fillStyle = bannerColor;
    ctx.fillRect(0, ch - barH, cw, barH);

    const fontSize = Math.max(16, Math.round(cw / 28));
    ctx.font      = `bold ${fontSize}px Inter, sans-serif`;
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(bannerText, cw / 2, ch - barH / 2);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
}

// ── Start / Stop ──────────────────────────────────────────────
async function startCamera() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });

        video.srcObject = webcamStream;
        video.style.display = 'block';
        await video.play();

        placeholder.style.display = 'none';
        scanOverlay.classList.add('active');

        document.getElementById('btn-start').disabled     = true;
        document.getElementById('btn-stop').disabled      = false;
        document.getElementById('btn-screenshot').disabled = false;

        updateCamStatus(true);
        showToast('Camera started', 'success');

        isDetecting = true;
        requestAnimationFrame(renderLoop);  // smooth overlay
        detectLoop();                       // background detection

    } catch (err) {
        showToast('Camera error: ' + err.message, 'error');
    }
}

function stopCamera() {
    isDetecting = false;
    clearTimeout(detectionTimer);
    lastResults = [];

    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }

    video.srcObject = null;
    video.style.display = 'none';
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    placeholder.style.display = 'flex';
    scanOverlay.classList.remove('active');

    document.getElementById('btn-start').disabled     = false;
    document.getElementById('btn-stop').disabled      = true;
    document.getElementById('btn-screenshot').disabled = true;

    updateCamStatus(false);
    const banner = document.getElementById('mask-status-banner');
    if (banner) banner.classList.add('hidden');
    showToast('Camera stopped', 'info');
}

// ── Update UI ─────────────────────────────────────────────────
function updateCamStatus(active) {
    const dot  = document.getElementById('cam-status');
    const text = document.getElementById('cam-status-text');
    if (active) {
        dot.classList.add('active');
        text.textContent = 'Live';
        text.style.color = 'var(--accent-success)';
    } else {
        dot.classList.remove('active');
        text.textContent = 'Camera Off';
        text.style.color = '';
    }
}

function updateStats(data, fps) {
    document.getElementById('stat-faces').textContent   = data.total_faces   || 0;
    document.getElementById('stat-masks').textContent   = data.mask_count    || 0;
    document.getElementById('stat-nomasks').textContent = data.no_mask_count || 0;
    document.getElementById('fps-display').textContent  = (fps || 0) + ' FPS';
}

function updateConfidence(detections) {
    if (!detections || detections.length === 0) {
        document.getElementById('confidence-value').textContent = '0%';
        document.getElementById('confidence-arc').setAttribute('stroke-dasharray', '0 251');
        return;
    }
    const avg = detections.reduce((s, d) => s + d.confidence, 0) / detections.length;
    document.getElementById('confidence-value').textContent = avg.toFixed(1) + '%';
    document.getElementById('confidence-arc').setAttribute('stroke-dasharray', `${(avg / 100) * 251} 251`);
}

function updateStatusBanner(data) {
    const banner = document.getElementById('mask-status-banner');
    const inner  = document.getElementById('banner-inner');
    const iconEl = document.getElementById('banner-icon');
    const textEl = document.getElementById('banner-text');
    if (!banner) return;

    if (!data.detections || data.detections.length === 0) {
        banner.classList.add('hidden');
        return;
    }
    const anyNoMask = data.detections.some(d => !d.has_mask);
    banner.classList.remove('hidden');
    if (anyNoMask) {
        inner.className    = 'banner-inner no-mask';
        iconEl.textContent = '✗';
        textEl.textContent = 'NO MASK FOUND';
    } else {
        inner.className    = 'banner-inner mask';
        iconEl.textContent = '✓';
        textEl.textContent = 'MASK FOUND';
    }
    updateConfidence(data.detections);
}

let lastAlertTime = 0;
function checkVoiceAlert(data) {
    if (!document.getElementById('voice-toggle')?.checked) return;
    if (!data.no_mask_count || data.no_mask_count === 0) return;
    const now = Date.now();
    if (now - lastAlertTime < 5000) return;
    lastAlertTime = now;
    const msg = new SpeechSynthesisUtterance('Warning! No mask detected!');
    msg.rate = 1.1; msg.volume = 0.8;
    speechSynthesis.speak(msg);
}

function addLogEntry(data) {
    const logList = document.getElementById('detection-log');
    const empty   = logList.querySelector('.log-empty');
    if (empty) empty.remove();

    if (!data.detections || data.detections.length === 0) return;

    data.detections.forEach(d => {
        const item = document.createElement('div');
        item.className = `log-item ${d.has_mask ? 'mask' : 'no-mask'}`;
        const time = new Date().toLocaleTimeString();
        item.innerHTML = `
            <i class="fas fa-${d.has_mask ? 'check-circle' : 'exclamation-triangle'}"
               style="color:${d.has_mask ? 'var(--accent-success)' : 'var(--accent-danger)'}"></i>
            <span>${d.label} (${d.confidence}%)</span>
            <span class="log-item-time">${time}</span>`;
        logList.prepend(item);
    });

    const items = logList.querySelectorAll('.log-item');
    for (let i = 20; i < items.length; i++) items[i].remove();
}

// ── Generate PDF Report ───────────────────────────────────────
async function generateReport() {
    const btn = document.getElementById('btn-report');
    const orig = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

    try {
        const res = await fetch('/api/report/pdf');
        if (!res.ok) throw new Error('Server error ' + res.status);

        const blob = await res.blob();
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `maskguard_report_${Date.now()}.pdf`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('PDF report downloaded!', 'success');
    } catch (err) {
        showToast('Failed to generate report: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = orig;
    }
}

// ── Screenshot ────────────────────────────────────────────────
function takeScreenshot() {
    const sc = document.createElement('canvas');
    sc.width  = video.videoWidth;
    sc.height = video.videoHeight;
    const ctx = sc.getContext('2d');
    ctx.drawImage(video,   0, 0);   // live video
    ctx.drawImage(overlay, 0, 0);   // detection boxes on top

    const link = document.createElement('a');
    link.download = `maskguard_${Date.now()}.jpg`;
    link.href = sc.toDataURL('image/jpeg', 0.9);
    link.click();
    showToast('Screenshot saved', 'success');
}
