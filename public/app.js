const fileInput = document.querySelector("#fileInput");
const chooseButton = document.querySelector("#chooseButton");
const changeButton = document.querySelector("#changeButton");
const correctButton = document.querySelector("#correctButton");
const dropzone = document.querySelector("#dropzone");
const editor = document.querySelector("#editor");
const status = document.querySelector("#status");
const fileName = document.querySelector("#fileName");
const CORRECTION_FONT_FAMILY = 'Kalam, "Comic Sans MS", cursive';
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
    transcriptButton: document.querySelector(`#${name}TranscriptButton`),
    processedImage: null,
    transcript: null,
    coordinateSpace: { width: 1000, height: 1000 }
  };
});

const transcriptOverlay = document.querySelector("#transcriptOverlay");
const transcriptText = document.querySelector("#transcriptText");
const transcriptPrintButton = document.querySelector("#transcriptPrintButton");
const transcriptCloseButton = document.querySelector("#transcriptCloseButton");
transcriptPrintButton.addEventListener("click", () => window.print());
transcriptCloseButton.addEventListener("click", () => { transcriptOverlay.hidden = true; });
transcriptOverlay.addEventListener("click", (event) => {
  if (event.target === transcriptOverlay) transcriptOverlay.hidden = true;
});

let sourceImage = null;
let sourceDataUrl = "";

chooseButton.addEventListener("click", () => fileInput.click());
changeButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => loadFile(fileInput.files[0]));
correctButton.addEventListener("click", correctLetter);
for (const provider of providers) provider.downloadButton.addEventListener("click", () => downloadResult(provider));
for (const provider of providers) provider.transcriptButton.addEventListener("click", () => showTranscript(provider));
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
    provider.transcript = null;
    provider.canvas.width = sourceImage.naturalWidth;
    provider.canvas.height = sourceImage.naturalHeight;
    drawOriginal(provider);
    provider.status.textContent = "Düzeltmeye hazır.";
    provider.status.classList.remove("error");
    provider.downloadButton.hidden = true;
    provider.transcriptButton.hidden = true;
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
    provider.transcript = null;
    drawOriginal(provider);
    provider.status.textContent = `${provider.label} metni kontrol ediyor…`;
    provider.status.classList.remove("error");
    provider.downloadButton.hidden = true;
    provider.transcriptButton.hidden = true;
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
    const languageToolNote = result.languagetool_ok === false
      ? ` (${result.languagetool_failures?.length || 1} cümlede dilbilgisi doğrulaması çalışmadı.)`
      : "";
    const unrenderableNote = result.unrenderable_corrections
      ? ` (${result.unrenderable_corrections} düzeltme fotoğrafta güvenle gösterilemediği için atlandı.)`
      : "";
    provider.status.textContent = (count
      ? `${count} düzeltme fotoğrafın üzerine işlendi.`
      : "Mektupta belirgin bir hata bulunamadı.") + fallbackNote + languageToolNote + unrenderableNote;
    provider.transcript = result.transcript || null;
    provider.transcriptButton.hidden = !provider.transcript?.length;
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
  // A physical line's own measured height can vary a fair bit across one
  // letter (handwriting naturally drifts bigger/smaller line to line), so
  // sizing a correction purely off its own line makes it stand out next to
  // neighbouring corrections even though nothing went wrong. Capping every
  // correction to a modest multiple of the letter's typical (median) line
  // height keeps the whole page visually consistent without hiding a
  // genuinely large line's need for a bigger label.
  const documentLineHeights = corrections
    .map((item) => (item.anchors || [])
      .map((anchor) => normalizedBoxToCanvas(engine, anchor).lineHeight)
      .filter(Number.isFinite))
    .flat()
    .sort((a, b) => a - b);
  const typicalLineHeight = documentLineHeights.length
    ? documentLineHeights[Math.floor(documentLineHeights.length / 2)]
    : undefined;
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
    const maximumFont = Math.max(12, canvas.width * .065);
    // A line whose own measured height happens to run bigger than the
    // letter's typical line (ordinary handwriting variance, not an error)
    // would otherwise size its correction well past every neighbouring one.
    const documentCappedLineHeight = typicalLineHeight
      ? Math.min(lineHeight, typicalLineHeight * 1.25)
      : lineHeight;
    let fontSize = Math.max(9, Math.min(
      documentCappedLineHeight * (item.action === "insert" ? .76 : .74),
      maximumFont
    ));

    ctx.save();
    ctx.fillStyle = "#d43f32";
    ctx.font = `700 ${fontSize}px ${CORRECTION_FONT_FAMILY}`;
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
    // Fitting a phrase into its slot has two knobs: shrink the font, or
    // squeeze it horizontally. Neither one, applied blindly, generalises: a
    // pure font shrink makes a short word look tiny next to full-size
    // neighbours, while a pure squeeze distorts letters once the required
    // ratio gets small. This applies uniformly to every correction
    // (replace/rewrite/insert, any word length) rather than special-casing
    // short words, since a short word can still land in a genuinely narrow
    // original gap and needs the same fallback as a long one.
    let textWidth = ctx.measureText(item.replacement).width;
    if (mayFitInsideSlot && textWidth > maximumTextWidth) {
      const naturalRequiredScale = maximumTextWidth / textWidth;
      const modestSqueezeFloor = .82;
      if (naturalRequiredScale < modestSqueezeFloor) {
        // A modest squeeze alone isn't enough - shrink the font toward a
        // readable floor first so the remaining squeeze stays mild. Fitting
        // inside the slot matters more than protecting this floor: two
        // adjacent words that both get corrected can each individually fit
        // their own original slot and still end up touching once both grow,
        // since neither one knows the other grew too. Letting the font
        // shrink further when a slot is essentially fully consumed keeps
        // the pair legible and separated instead of overlapping.
        const minimumReadableFont = Math.max(6, lineHeight * .38);
        fontSize = Math.max(minimumReadableFont, fontSize * naturalRequiredScale);
        ctx.font = `700 ${fontSize}px ${CORRECTION_FONT_FAMILY}`;
        textWidth = ctx.measureText(item.replacement).width;
      }
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
    // The font shrink above already does most of the work, so this floor
    // only matters for whatever residual squeeze remains. Keep it low
    // enough that a slot with almost no room left (two touching corrected
    // words) still ends up fitting rather than overlapping its neighbour.
    const horizontalScale = mayFitInsideSlot && !placeAbove && !placeBelow
      ? Math.min(1, Math.max(.66, requiredScale))
      : 1;
    const paintedWidth = textWidth * horizontalScale;
    const labelX = item.action === "insert" ? x - paintedWidth / 2 : x + (width - paintedWidth) / 2;
    const rawLabelY = placeAbove
      ? Math.max(placement.safeTop + fontSize, y - fontSize * .12)
      : placeBelow
        ? Math.min(placement.safeBottom, placement.baseline + fontSize * 1.05)
        : item.action === "insert" && placement.between
      ? placement.baseline - height * .04
      : (item.action === "replace" || item.action === "rewrite_line")
        ? (placement.lineBaseline || (y + height)) - lineHeight * .025
        : Math.max(fontSize + 3, y + height * .08);
    // A rewrite that merges anchors from two physical lines (a sentence
    // group can now span line breaks) averages their baselines in
    // getRewritePlacement, which can land the label between the two lines
    // instead of on either one. Clamp every label inside its own safe
    // vertical band so it never bleeds into a neighbouring line.
    const labelY = Number.isFinite(placement.safeTop) && Number.isFinite(placement.safeBottom)
      ? Math.min(placement.safeBottom, Math.max(placement.safeTop + fontSize * .8, rawLabelY))
      : rawLabelY;
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
      document.fonts.load(`700 32px ${CORRECTION_FONT_FAMILY}`),
      document.fonts.ready
    ]).catch(() => undefined);
  }
  return correctionFontReady;
}

function getRewritePlacement(anchors) {
  // A phrase owns the whitespace slots around all of its erased OCR words.
  // This gives replacements enough room without borrowing space from a
  // different physical line or centring them over an unrelated neighbour.
  const tightLeft = Math.min(...anchors.map((anchor) => anchor.x));
  const tightRight = Math.max(...anchors.map((anchor) => anchor.x + anchor.width));
  const borrowedLeft = Math.min(...anchors.map((anchor) => Number.isFinite(anchor.slotX) ? anchor.slotX : anchor.x));
  const borrowedRight = Math.max(...anchors.map((anchor) => Number.isFinite(anchor.slotX) && Number.isFinite(anchor.slotWidth)
    ? anchor.slotX + anchor.slotWidth
    : anchor.x + anchor.width));
  const top = Math.min(...anchors.map((anchor) => anchor.y));
  const bottom = Math.max(...anchors.map((anchor) => anchor.y + anchor.height));
  const baselines = anchors.map((anchor) => anchor.lineBaseline).filter(Number.isFinite);
  const lineHeights = anchors.map((anchor) => anchor.lineHeight).filter(Number.isFinite);
  // The first/last word on a physical line has no real neighbour on its
  // outer side, so its slot is synthesised as if a same-sized word sat
  // there - up to half that word's own width of make-believe whitespace.
  // Borrowing real neighbouring whitespace is fine, but how far the phrase
  // may drift past its own tight bounds should be capped by how big the
  // handwriting itself is here (its line height), not by how long this one
  // replacement phrase happens to be - a one-word and a five-word rewrite
  // on the same line sit in the same natural margin, so they should get the
  // same cushion.
  const cushion = (averageFinite(lineHeights) || (bottom - top)) * 1.1;
  const left = Math.max(borrowedLeft, tightLeft - cushion);
  const right = Math.min(borrowedRight, tightRight + cushion);
  const safeTops = anchors.map((anchor) => anchor.safeTop).filter(Number.isFinite);
  const safeBottoms = anchors.map((anchor) => anchor.safeBottom).filter(Number.isFinite);
  return {
    x: left,
    y: top,
    width: right - left,
    height: bottom - top,
    lineBaseline: averageFinite(baselines) || bottom,
    lineHeight: averageFinite(lineHeights) || bottom - top,
    // All target anchors of a rewrite share one physical line, so take the
    // widest shared band rather than averaging - a rewrite spanning two
    // merged sentence-group lines must still stay clamped inside it.
    safeTop: safeTops.length ? Math.min(...safeTops) : undefined,
    safeBottom: safeBottoms.length ? Math.max(...safeBottoms) : undefined
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

function showTranscript(provider) {
  if (!provider.transcript?.length) return;
  transcriptText.innerHTML = provider.transcript
    .map((token) => token.changed
      ? `<span class="fixed">${escapeHtml(token.text)}</span>`
      : escapeHtml(token.text))
    .join(" ");
  transcriptOverlay.hidden = false;
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
