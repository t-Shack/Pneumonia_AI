const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const lightbox = document.getElementById('lightbox');
const xrayImage = document.getElementById('xray-image');
const scanline = document.getElementById('scanline');
const errorMessage = document.getElementById('error-message');

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.hidden = true;
  errorMessage.textContent = '';
}

function labelClass(predictedClass) {
  return predictedClass === 'PNEUMONIA' ? 'pneumonia' : 'normal';
}

function renderReadout(prefix, result) {
  const cls = labelClass(result.predicted_class);
  const confidencePct = (result.confidence * 100).toFixed(1);

  const verdictEl = document.getElementById(`verdict-${prefix}`);
  verdictEl.textContent = result.predicted_class;
  verdictEl.className = `verdict ${cls}`;

  document.getElementById(`confidence-${prefix}`).textContent =
    `${confidencePct}% confidence`;

  const gaugeEl = document.getElementById(`gauge-${prefix}`);
  gaugeEl.className = `gauge-fill ${cls}`;
  gaugeEl.style.width = '0%';
  requestAnimationFrame(() => { gaugeEl.style.width = `${confidencePct}%`; });

  const probsEl = document.getElementById(`probs-${prefix}`);
  probsEl.innerHTML = '';
  Object.entries(result.probabilities).forEach(([label, prob]) => {
    const row = document.createElement('div');
    row.textContent = `${label.padEnd(10, ' ')} ${(prob * 100).toFixed(1)}%`;
    probsEl.appendChild(row);
  });
}

async function handleFile(file) {
  clearError();

  if (!/\.(png|jpe?g)$/i.test(file.name)) {
    showError('Use a PNG or JPEG image.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError('File is larger than 10MB.');
    return;
  }

  lightbox.hidden = false;
  scanline.classList.add('active');
  xrayImage.src = URL.createObjectURL(file);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json();

    if (!response.ok) {
      showError(data.error || 'Something went wrong.');
      scanline.classList.remove('active');
      return;
    }

    renderReadout('sigmoid', data.results.sigmoid);
    renderReadout('softmax', data.results.softmax);
  } catch (err) {
    showError('Could not reach the server.');
  } finally {
    scanline.classList.remove('active');
  }
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

['dragenter', 'dragover'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  })
);

['dragleave', 'drop'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  })
);

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
