const fileInput = document.querySelector("#fileInput");
const chooseButton = document.querySelector("#chooseButton");
const changeButton = document.querySelector("#changeButton");
const correctButton = document.querySelector("#correctButton");
const dropzone = document.querySelector("#dropzone");
const editor = document.querySelector("#editor");
const status = document.querySelector("#status");
const fileName = document.querySelector("#fileName");
const CORRECTION_FONT_FAMILY = 'Caveat, "Comic Sans MS", cursive';
let correctionFontReady = null;

const providers = ["gemini"].map((name) => {
  const canvas = document.querySelector(`#${name}Canvas`);
  return {
    name,
    label: "Google Gemini",
    canvas,
    ctx: canvas.getContext("2d"),
    status: document.querySelector(`#${name}Status`),
    correctionList: document.querySelector(`#${name}CorrectionList`),
    downloadButton: document.querySelector(`#${name}DownloadButton`),
    processedImage: null,
    coordinateSpace: { width: 1000, height: 1000 }
  };
});

let sourceImage = null;
let sourceDataUrl = "";

chooseButton.addEventListener("click", () => fileInput.click());
changeButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => loadFile(fileInput.files[0]));
correctButton.addEventListener("click", correctLetter);
for (const provider of providers) provider.downloadButton.addEventListener("click", () => downloadResult(provider));
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") fileInput.click();
});

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}
dropzone.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));

async function loadFile(file) {
  if (!file) return;
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)) return showStatus("JPG, PNG veya WebP seçmelisin.", true);
  if (file.size > 12 * 1024 * 1024) return showStatus("Fotoğraf 12 MB'dan küçük olmalı.", true);

  sourceDataUrl = await readFile(file);
  sourceImage = await loadImage(sourceDataUrl);
  for (const provider of providers) {
    provider.processedImage = null;
    provider.canvas.width = sourceImage.naturalWidth;
    provider.canvas.height = sourceImage.naturalHeight;
    drawOriginal(provider);
    provider.status.textContent = "Düzeltmeye hazır.";
    provider.status.classList.remove("error");
    provider.downloadButton.hidden = true;
    provider.correctionList.hidden = true;
  }

  fileName.textContent = file.name;
  dropzone.hidden = true;
  editor.hidden = false;
  correctButton.hidden = false;
  showStatus("Google OCR metni Gemini ile kontrol edilecek.");
}

async function correctLetter() {
  if (!sourceDataUrl) return;
  correctButton.disabled = true;
  showStatus("Google OCR okunuyor; Gemini metni kontrol ediyor…");
  for (const provider of providers) {
    provider.processedImage = null;
    drawOriginal(provider);
    provider.status.textContent = `${provider.label} metni kontrol ediyor…`;
    provider.status.classList.remove("error");
    provider.downloadButton.hidden = true;
    provider.correctionList.hidden = true;
  }

  const successful = await runProvider(providers[0]);
  showStatus(successful ? "Gemini sonucu hazır." : "Gemini sonuç üretemedi.", !successful);
  correctButton.disabled = false;
}

async function runProvider(provider) {
  try {
    const response = await fetch("/api/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: sourceDataUrl, ai_provider: provider.name })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Düzeltme yapılamadı.");

    provider.coordinateSpace = result.coordinate_space || provider.coordinateSpace;
    if (result.cleaned_image) {
      provider.processedImage = await loadImage(result.cleaned_image);
      provider.canvas.width = provider.processedImage.naturalWidth;
      provider.canvas.height = provider.processedImage.naturalHeight;
    }
    // Canvas does not repaint text when a web font finishes loading. Always
    // settle the correction font before the first annotation so Gemini and
    // fallback results cannot accidentally use different font families.
    await ensureCorrectionFont();
    drawOriginal(provider);
    drawCorrections(provider, result.corrections);
    renderCorrectionList(provider, result.corrections);
    const count = result.corrections.length;
    const fallbackNote = result.analysis_source === "gemini_with_fallback"
      ? ` (${result.fallback_groups?.length || 1} cümlede güvenli yedek kullanıldı.)`
      : "";
    provider.status.textContent = (count
      ? `${count} düzeltme fotoğrafın üzerine işlendi.`
      : "Mektupta belirgin bir hata bulunamadı.") + fallbackNote;
    provider.downloadButton.hidden = false;
    return true;
  } catch (error) {
    provider.status.textContent = error.message;
    provider.status.classList.add("error");
    return false;
  }
}

function drawOriginal(engine) {
  engine.ctx.clearRect(0, 0, engine.canvas.width, engine.canvas.height);
  engine.ctx.drawImage(engine.processedImage || sourceImage, 0, 0);
}

function drawCorrections(engine, corrections) {
  const { ctx, canvas } = engine;
  for (const item of corrections) {
    const anchors = (item.anchors || []).map((anchor) => normalizedBoxToCanvas(engine, anchor));
    if (!anchors.length) continue;
    const placement = item.action === "insert"
      ? getInsertionPlacement(anchors)
      : item.action === "rewrite_line"
        ? getRewritePlacement(anchors)
        : anchors.find((anchor) => anchor.relation === "target") || anchors[0];
    let { x, y, width, height } = placement;
    if (item.action === "replace" && Number.isFinite(placement.slotX) && Number.isFinite(placement.slotWidth)) {
      // Neighbouring whitespace may be borrowed, but the replacement must
      // remain centred on the erased word. Using the entire asymmetric slot
      // was the reason words such as “movies” visibly jumped sideways.
      const targetCenter = placement.x + placement.width / 2;
      const slotLeft = placement.slotX;
      const slotRight = placement.slotX + placement.slotWidth;
      const symmetricRoom = Math.max(
        placement.width / 2,
        Math.min(targetCenter - slotLeft, slotRight - targetCenter)
      );
      x = targetCenter - symmetricRoom;
      width = symmetricRoom * 2;
    }
    const lineHeight = placement.lineHeight || height;
    // Size annotations from the OCR line in the *processed* image. Using the
    // canvas' short edge here made text tiny after a wide, single-line photo
    // was cropped (for example 906x426 capped every label near 16px).
    const maximumFont = Math.max(12, canvas.width * .075);
    let fontSize = Math.max(9, Math.min(
      lineHeight * (item.action === "insert" ? .86 : .84),
      maximumFont
    ));

    ctx.save();
    ctx.fillStyle = "#d43f32";
    ctx.font = `600 ${fontSize}px ${CORRECTION_FONT_FAMILY}`;
    // Fit phrases by reducing the font modestly. Never stretch short words and
    // never apply OCR polygon angles: handwriting polygons are too noisy for
    // reliable rotation, especially on a curved sheet.
    ctx.textBaseline = "bottom";
    ctx.shadowColor = "rgba(130, 25, 18, .12)";
    ctx.shadowBlur = 1;
    const maximumTextWidth = Math.max(1, width * .96);
    const mayFitInsideSlot = item.action === "replace"
      || item.action === "rewrite_line"
      || (item.action === "insert" && placement.between);
    const compactInsertion = item.action === "insert"
      && !/\s/.test(item.replacement.trim())
      && item.replacement.trim().length <= 4;
    let textWidth = ctx.measureText(item.replacement).width;
    if (mayFitInsideSlot && textWidth > maximumTextWidth && !compactInsertion) {
      const minimumReadableFont = Math.max(7, lineHeight * .54);
      fontSize = Math.max(minimumReadableFont, fontSize * maximumTextWidth / textWidth);
      ctx.font = `600 ${fontSize}px ${CORRECTION_FONT_FAMILY}`;
      textWidth = ctx.measureText(item.replacement).width;
    }
    const requiredScale = maximumTextWidth / Math.max(1, textWidth);
    const roomAbove = Number.isFinite(placement.safeTop) ? y - placement.safeTop : 0;
    const roomBelow = Number.isFinite(placement.safeBottom)
      ? placement.safeBottom - placement.baseline
      : 0;
    // “is/the/to” gibi dar aralığa eklenen kelimeleri okunmayacak kadar yatay
    // sıkıştırmak yerine, aynı satırın güvenli boşluğuna tam ölçekte taşı.
    const placeAbove = item.action === "insert" && placement.between
      && requiredScale < .72 && roomAbove >= fontSize * .9;
    const placeBelow = item.action === "insert" && placement.between
      && !placeAbove && requiredScale < .72 && roomBelow >= fontSize * 1.05;
    const horizontalScale = mayFitInsideSlot && !placeAbove && !placeBelow
      ? Math.min(1, requiredScale)
      : 1;
    const paintedWidth = textWidth * horizontalScale;
    const labelX = item.action === "insert" ? x - paintedWidth / 2 : x + (width - paintedWidth) / 2;
    const labelY = placeAbove
      ? Math.max(placement.safeTop + fontSize, y - fontSize * .12)
      : placeBelow
        ? Math.min(placement.safeBottom, placement.baseline + fontSize * 1.05)
        : item.action === "insert" && placement.between
      ? placement.baseline - height * .04
      : (item.action === "replace" || item.action === "rewrite_line")
        ? (placement.lineBaseline || (y + height)) - lineHeight * .025
        : Math.max(fontSize + 3, y + height * .08);
    ctx.translate(Math.max(2, labelX), labelY);
    ctx.scale(horizontalScale, 1);
    ctx.fillText(item.replacement, 0, 0);
    ctx.restore();
  }
}

function ensureCorrectionFont() {
  if (!document.fonts?.load) return Promise.resolve();
  if (!correctionFontReady) {
    correctionFontReady = Promise.all([
      document.fonts.load(`600 32px ${CORRECTION_FONT_FAMILY}`),
      document.fonts.ready
    ]).catch(() => undefined);
  }
  return correctionFontReady;
}

function getRewritePlacement(anchors) {
  // A phrase owns the whitespace slots around all of its erased OCR words.
  // This gives replacements enough room without borrowing space from a
  // different physical line or centring them over an unrelated neighbour.
  const left = Math.min(...anchors.map((anchor) => Number.isFinite(anchor.slotX) ? anchor.slotX : anchor.x));
  const top = Math.min(...anchors.map((anchor) => anchor.y));
  const right = Math.max(...anchors.map((anchor) => Number.isFinite(anchor.slotX) && Number.isFinite(anchor.slotWidth)
    ? anchor.slotX + anchor.slotWidth
    : anchor.x + anchor.width));
  const bottom = Math.max(...anchors.map((anchor) => anchor.y + anchor.height));
  const baselines = anchors.map((anchor) => anchor.lineBaseline).filter(Number.isFinite);
  const lineHeights = anchors.map((anchor) => anchor.lineHeight).filter(Number.isFinite);
  return {
    x: left,
    y: top,
    width: right - left,
    height: bottom - top,
    lineBaseline: averageFinite(baselines) || bottom,
    lineHeight: averageFinite(lineHeights) || bottom - top
  };
}

function normalizedBoxToCanvas(engine, anchor) {
  return {
    ...anchor,
    x: anchor.x * engine.canvas.width / engine.coordinateSpace.width,
    y: anchor.y * engine.canvas.height / engine.coordinateSpace.height,
    width: anchor.width * engine.canvas.width / engine.coordinateSpace.width,
    height: anchor.height * engine.canvas.height / engine.coordinateSpace.height,
    lineBaseline: Number.isFinite(anchor.lineBaseline)
      ? anchor.lineBaseline * engine.canvas.height / engine.coordinateSpace.height
      : undefined,
    lineHeight: Number.isFinite(anchor.lineHeight)
      ? anchor.lineHeight * engine.canvas.height / engine.coordinateSpace.height
      : undefined,
    slotX: Number.isFinite(anchor.slotX)
      ? anchor.slotX * engine.canvas.width / engine.coordinateSpace.width
      : undefined,
    slotWidth: Number.isFinite(anchor.slotWidth)
      ? anchor.slotWidth * engine.canvas.width / engine.coordinateSpace.width
      : undefined,
    safeTop: Number.isFinite(anchor.safeTop)
      ? anchor.safeTop * engine.canvas.height / engine.coordinateSpace.height
      : undefined,
    safeBottom: Number.isFinite(anchor.safeBottom)
      ? anchor.safeBottom * engine.canvas.height / engine.coordinateSpace.height
      : undefined
  };
}

function getInsertionPlacement(anchors) {
  const left = anchors.find((anchor) => anchor.relation === "left");
  const right = anchors.find((anchor) => anchor.relation === "right");
  if (left && right) {
    const leftEdge = left.x + left.width;
    const rightEdge = right.x;
    return {
      x: (leftEdge + rightEdge) / 2,
      y: (left.y + right.y) / 2,
      width: Math.max(1, rightEdge - leftEdge),
      height: (left.height + right.height) / 2,
      baseline: averageFinite([left.lineBaseline, right.lineBaseline])
        || ((left.y + left.height) + (right.y + right.height)) / 2,
      lineBaseline: averageFinite([left.lineBaseline, right.lineBaseline]),
      lineHeight: averageFinite([left.lineHeight, right.lineHeight]),
      safeTop: averageFinite([left.safeTop, right.safeTop]),
      safeBottom: averageFinite([left.safeBottom, right.safeBottom]),
      between: true
    };
  }
  const anchor = left || right || anchors[0];
  return {
    x: anchor.relation === "left" ? anchor.x + anchor.width : anchor.x,
    y: anchor.y,
    width: Math.max(1, anchor.width * .2),
    height: anchor.height,
    baseline: anchor.y + anchor.height,
    safeTop: anchor.safeTop,
    safeBottom: anchor.safeBottom,
    between: false
  };
}

function averageFinite(values) {
  const finite = values.filter(Number.isFinite);
  return finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : undefined;
}

function renderCorrectionList(engine, corrections) {
  engine.correctionList.innerHTML = corrections.map((item) => `
    <div class="correction-item">
      <span class="wrong">${item.action === "insert" ? "Eksik" : escapeHtml(item.original)}</span>
      <span>→</span>
      <span class="right">${escapeHtml(item.replacement)}</span>
      <span class="reason">${escapeHtml(item.reason)}</span>
    </div>`).join("");
  engine.correctionList.hidden = corrections.length === 0;
}

function downloadResult(engine) {
  const link = document.createElement("a");
  link.download = `duzeltilmis-mektup-${engine.name}.png`;
  link.href = engine.canvas.toDataURL("image/png", 1);
  link.click();
}

function showStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = url;
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[char]);
}
