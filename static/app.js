const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("statusText");
const transcriptEl = document.getElementById("transcript");
const assistantTextEl = document.getElementById("assistantText");
const player = document.getElementById("player");

let mediaRecorder;
let chunks = [];

recordBtn.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    chunks = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "audio/webm" });
      await uploadAudio(blob);
      stream.getTracks().forEach((track) => track.stop());
    };

    mediaRecorder.start();
    setStatus("Recording...");
    recordBtn.disabled = true;
    stopBtn.disabled = false;
  } catch (error) {
    setStatus(`Microphone access failed: ${error.message}`);
  }
});

stopBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    setStatus("Uploading audio...");
  }
  stopBtn.disabled = true;
  recordBtn.disabled = false;
});

async function uploadAudio(blob) {
  const form = new FormData();
  form.append("file", blob, "recording.webm");
  form.append("mode", "default");

  try {
    setStatus("Processing...");
    const response = await fetch("/api/voice", { method: "POST", body: form });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Voice processing failed");
    }

    transcriptEl.textContent = data.transcript || "-";
    assistantTextEl.textContent = data.assistant_text || "-";
    player.src = `/api/jobs/${data.job_id}/audio`;
    player.load();

    setStatus(`Done. Job ID: ${data.job_id}`);
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  }
}

function setStatus(text) {
  statusText.textContent = text;
}
