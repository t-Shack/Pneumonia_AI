const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const dropzoneInner = document.getElementById('dropzone-inner');
const analyzeBtn = document.getElementById('analyze-btn');
const uploadForm = document.getElementById('upload-form');

function handleFile(file) {
  if (!file) return;
  if (!/\.(png|jpe?g)$/i.test(file.name)) {
    alert('Please use a PNG or JPEG image.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    alert('File is larger than 10MB.');
    return;
  }

  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;

  previewImage.src = URL.createObjectURL(file);
  previewImage.hidden = false;
  dropzoneInner.hidden = true;
  analyzeBtn.disabled = false;
}

if (dropzone) {
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
}

if (uploadForm) {
  uploadForm.addEventListener('submit', () => {
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing…';
  });
}
