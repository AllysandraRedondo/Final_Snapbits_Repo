let remaining = 0;
let timerInterval = null;
let warningShown = false; // ✅ ensures half-time warning triggers only once
let halfTime = 0; // store half of initial remaining time

// Fetch current remaining time from backend
async function fetchInitialTime() {
  try {
    const res = await fetch("/get_remaining_time");
    const data = await res.json();
    remaining = data.remaining || 0;

    // Calculate half-time
    halfTime = Math.floor(remaining / 2);
  } catch (err) {
    console.error("Error fetching remaining time:", err);
    remaining = 0;
    halfTime = 0;
  }
}

// Save current preview / canvas to /save_photo
async function saveCurrentPhoto() {
  const previewContainer = document.getElementById("previewContainer");
  if (!previewContainer) return;

  const canvas = await html2canvas(previewContainer, {
    backgroundColor: null,
    scale: 3,
  });

  const flippedCanvas = document.createElement("canvas");
  flippedCanvas.width = canvas.width;
  flippedCanvas.height = canvas.height;
  const ctx = flippedCanvas.getContext("2d");

  // Flip horizontally
  ctx.translate(flippedCanvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(canvas, 0, 0);

  const imageData = flippedCanvas.toDataURL("image/png");

  await fetch("/save_photo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageData }),
  });

  console.log("✅ Preview saved in high quality, flipped horizontally");
}

// Timer tick
function tick() {
  const mins = Math.floor(remaining / 60);
  const secs = remaining % 60;
  const timerEl = document.getElementById("timer");
  if (timerEl) {
    timerEl.innerText = `${mins}:${secs.toString().padStart(2, "0")}`;
  }

  // ✅ Half-time warning
  if (!warningShown && remaining <= halfTime && remaining > 0) {
    warningShown = true;

    // Change timer style
    if (timerEl) {
      timerEl.style.backgroundColor = "red";
      timerEl.style.color = "black";
    }

    // Show warning popup
    const popup = document.getElementById("warningPopup");
    const sound = document.getElementById("warningSound");
    if (popup && sound) {
      popup.style.display = "flex";
      sound.play();
      setTimeout(() => {
        popup.style.display = "none";
      }, 2000);
    }
  }

  if (remaining <= 0) {
    clearInterval(timerInterval);
    saveCurrentPhoto().then(() => {
      window.location.href = "/numberOfCopies";
    });
    return;
  }

  remaining--;
}

// Start timer
async function startTimer() {
  await fetchInitialTime();
  tick(); // initial display immediately
  timerInterval = setInterval(tick, 1000);
}

startTimer();
