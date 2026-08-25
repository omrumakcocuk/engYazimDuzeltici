const fileInput = document.querySelector("#fileInput");
const chooseButton = document.querySelector("#chooseButton");
const changeButton = document.querySelector("#changeButton");
const correctButton = document.querySelector("#correctButton");
const downloadButton = document.querySelector("#downloadButton");
const dropzone = document.querySelector("#dropzone");
const editor = document.querySelector("#editor");
const canvas = document.querySelector("#canvas");
const ctx = canvas.getContext("2d");
const status = document.querySelector("#status");
const correctionList = document.querySelector("#correctionList");
const fileName = document.querySelector("#fileName");

let sourceImage = null;
let processedImage = null;
let sourceDataUrl = "";
let coordinateSpace = { width: 1000, height: 1000 };

chooseButton.addEventListener("click", () => fileInput.click());
changeButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => loadFile(fileInput.files[0]));
correctButton.addEventListener("click", correctLetter);
downloadButton.addEventListener("click", downloadResult);
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
  processedImage = null;
  canvas.width = sourceImage.naturalWidth;
  canvas.height = sourceImage.naturalHeight;
  drawOriginal();

  fileName.textContent = file.name;
  dropzone.hidden = true;
  editor.hidden = false;
  correctButton.hidden = false;
  downloadButton.hidden = true;
  correctionList.hidden = true;
  showStatus("Fotoğrafı kontrol etmeye hazırız.");
}

async function correctLetter() {
  if (!sourceDataUrl) return;
  correctButton.disabled = true;
  downloadButton.hidden = true;
  correctionList.hidden = true;
  processedImage = null;
  drawOriginal();
  showStatus("El yazısı okunuyor ve İngilizce kontrol ediliyor…");

  try {
    const response = await fetch("/api/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: sourceDataUrl })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Düzeltme yapılamadı.");

    coordinateSpace = result.coordinate_space || coordinateSpace;
    if (result.cleaned_image) processedImage = await loadImage(result.cleaned_image);
    drawOriginal();
    drawCorrections(result.corrections);
    renderCorrectionList(result.corrections);
    const count = result.corrections.length;
    showStatus(count ? `${count} düzeltme fotoğrafın üzerine işlendi.` : "Mektupta belirgin bir hata bulunamadı.");
    downloadButton.hidden = false;
  } catch (error) {
    showStatus(error.message, true);
  } finally {
    correctButton.disabled = false;
  }
}

function drawOriginal() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(processedImage || sourceImage, 0, 0);
}

function drawCorrections(corrections) {
  for (const item of corrections) {
    const anchors = (item.anchors || []).map(normalizedBoxToCanvas);
    if (!anchors.length) continue;

    const placement = item.action === "insert"
      ? getInsertionPlacement(anchors)
      : anchors.find((anchor) => anchor.relation === "target") || anchors[0];
    const { x, y, width, height } = placement;
    let fontSize = Math.max(18, Math.min(
      height * (item.action === "insert" ? .70 : .86),
      canvas.width * .055
    ));

    ctx.save();
    ctx.strokeStyle = "#d43f32";
    ctx.fillStyle = "#d43f32";
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(2, fontSize * .065);

    ctx.font = `600 ${fontSize}px Caveat, "Comic Sans MS", cursive`;
    if (item.action === "replace") {
      const initialWidth = ctx.measureText(item.replacement).width;
      const maximumWidth = Math.max(width * 1.08, 1);
      if (initialWidth > maximumWidth) {
        fontSize = Math.max(12, fontSize * maximumWidth / initialWidth);
        ctx.lineWidth = Math.max(2, fontSize * .065);
        ctx.font = `600 ${fontSize}px Caveat, "Comic Sans MS", cursive`;
      }
    }
    ctx.textBaseline = "bottom";
    ctx.shadowColor = "rgba(130, 25, 18, .12)";
    ctx.shadowBlur = 1;
    const textWidth = ctx.measureText(item.replacement).width;
    const isShortInsertion = item.action === "insert"
      && item.replacement.trim().length <= 3;
    const fitsInline = item.action === "insert"
      && (!placement.between || isShortInsertion || textWidth + fontSize * .15 <= width);
    const labelX = item.action === "insert"
      ? x - textWidth / 2
      : x + (width - textWidth) / 2;
    const labelY = item.action === "insert" && fitsInline
      ? placement.baseline - height * .04
      : item.action === "replace"
        ? y + height * .94
        : Math.max(fontSize + 3, y - height * .12);
    ctx.fillText(item.replacement, Math.max(2, labelX), labelY);
    ctx.restore();
  }
}

function normalizedBoxToCanvas(anchor) {
  const scaleX = canvas.width / coordinateSpace.width;
  const scaleY = canvas.height / coordinateSpace.height;
  return {
    ...anchor,
    x: anchor.x * scaleX,
    y: anchor.y * scaleY,
    width: anchor.width * scaleX,
    height: anchor.height * scaleY
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

function renderCorrectionList(corrections) {
  correctionList.innerHTML = corrections.map((item) => `
    <div class="correction-item">
      <span class="wrong">${item.action === "insert" ? "Eksik" : escapeHtml(item.original)}</span>
      <span>→</span>
      <span class="right">${escapeHtml(item.replacement)}</span>
      <span class="reason">${escapeHtml(item.reason)}</span>
    </div>`).join("");
  correctionList.hidden = corrections.length === 0;
}

function downloadResult() {
  const link = document.createElement("a");
  link.download = "duzeltilmis-mektup.png";
  link.href = canvas.toDataURL("image/png", 1);
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
