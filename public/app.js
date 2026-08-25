const fileInput = document.querySelector("#fileInput");
const chooseButton = document.querySelector("#chooseButton");
const changeButton = document.querySelector("#changeButton");
const correctButton = document.querySelector("#correctButton");
const dropzone = document.querySelector("#dropzone");
const editor = document.querySelector("#editor");
const status = document.querySelector("#status");
const fileName = document.querySelector("#fileName");

const engines = ["apple", "google"].map((name) => {
  const canvas = document.querySelector(`#${name}Canvas`);
  return {
    name,
    label: name === "apple" ? "Apple Vision" : "Google Cloud Vision",
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
for (const engine of engines) engine.downloadButton.addEventListener("click", () => downloadResult(engine));
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
  for (const engine of engines) {
    engine.processedImage = null;
    engine.canvas.width = sourceImage.naturalWidth;
    engine.canvas.height = sourceImage.naturalHeight;
    drawOriginal(engine);
    engine.status.textContent = "Düzeltmeye hazır.";
    engine.status.classList.remove("error");
    engine.downloadButton.hidden = true;
    engine.correctionList.hidden = true;
  }

  fileName.textContent = file.name;
  dropzone.hidden = true;
  editor.hidden = false;
  correctButton.hidden = false;
  showStatus("Aynı fotoğraf Apple ve Google OCR ile karşılaştırılacak.");
}

async function correctLetter() {
  if (!sourceDataUrl) return;
  correctButton.disabled = true;
  showStatus("İki OCR motoru paralel olarak çalışıyor…");
  for (const engine of engines) {
    engine.processedImage = null;
    drawOriginal(engine);
    engine.status.textContent = `${engine.label} el yazısını okuyor…`;
    engine.status.classList.remove("error");
    engine.downloadButton.hidden = true;
    engine.correctionList.hidden = true;
  }

  const outcomes = await Promise.all(engines.map(runEngine));
  const successful = outcomes.filter(Boolean).length;
  showStatus(successful === 2
    ? "İki sonuç hazır; yan yana karşılaştırabilirsin."
    : successful === 1
      ? "Bir OCR sonucu hazır; diğer paneldeki kurulum/hata mesajını kontrol et."
      : "İki OCR motoru da sonuç üretemedi.", successful === 0);
  correctButton.disabled = false;
}

async function runEngine(engine) {
  try {
    const response = await fetch("/api/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: sourceDataUrl, ocr_engine: engine.name })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Düzeltme yapılamadı.");

    engine.coordinateSpace = result.coordinate_space || engine.coordinateSpace;
    if (result.cleaned_image) engine.processedImage = await loadImage(result.cleaned_image);
    drawOriginal(engine);
    drawCorrections(engine, result.corrections);
    renderCorrectionList(engine, result.corrections);
    const count = result.corrections.length;
    engine.status.textContent = count
      ? `${count} düzeltme fotoğrafın üzerine işlendi.`
      : "Mektupta belirgin bir hata bulunamadı.";
    engine.downloadButton.hidden = false;
    return true;
  } catch (error) {
    engine.status.textContent = error.message;
    engine.status.classList.add("error");
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
      : anchors.find((anchor) => anchor.relation === "target") || anchors[0];
    const { x, y, width, height } = placement;
    let fontSize = Math.max(18, Math.min(height * (item.action === "insert" ? .70 : .86), canvas.width * .055));

    ctx.save();
    ctx.fillStyle = "#d43f32";
    ctx.font = `600 ${fontSize}px Caveat, "Comic Sans MS", cursive`;
    if (item.action === "replace") {
      const initialWidth = ctx.measureText(item.replacement).width;
      const maximumWidth = Math.max(width * 1.08, 1);
      if (initialWidth > maximumWidth) {
        fontSize = Math.max(12, fontSize * maximumWidth / initialWidth);
        ctx.font = `600 ${fontSize}px Caveat, "Comic Sans MS", cursive`;
      }
    }
    ctx.textBaseline = "bottom";
    ctx.shadowColor = "rgba(130, 25, 18, .12)";
    ctx.shadowBlur = 1;
    const textWidth = ctx.measureText(item.replacement).width;
    const isShortInsertion = item.action === "insert" && item.replacement.trim().length <= 3;
    const fitsInline = item.action === "insert" && (!placement.between || isShortInsertion || textWidth + fontSize * .15 <= width);
    const labelX = item.action === "insert" ? x - textWidth / 2 : x + (width - textWidth) / 2;
    const labelY = item.action === "insert" && fitsInline
      ? placement.baseline - height * .04
      : item.action === "replace" ? y + height * .94 : Math.max(fontSize + 3, y - height * .12);
    ctx.fillText(item.replacement, Math.max(2, labelX), labelY);
    ctx.restore();
  }
}

function normalizedBoxToCanvas(engine, anchor) {
  return {
    ...anchor,
    x: anchor.x * engine.canvas.width / engine.coordinateSpace.width,
    y: anchor.y * engine.canvas.height / engine.coordinateSpace.height,
    width: anchor.width * engine.canvas.width / engine.coordinateSpace.width,
    height: anchor.height * engine.canvas.height / engine.coordinateSpace.height
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
      baseline: ((left.y + left.height) + (right.y + right.height)) / 2,
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
    between: false
  };
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
