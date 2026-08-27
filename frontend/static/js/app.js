const state = {
    currentMode: 'image',
    selectedFile: null,
    selectedType: 'image',
    currentResult: null,
};

const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const uploadText = document.getElementById('uploadText');
const fileHint = document.getElementById('fileHint');
const uploadPrompt = document.getElementById('uploadPrompt');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const videoPreview = document.getElementById('videoPreview');
const btnAnalyze = document.getElementById('btnAnalyze');
const loadingBox = document.getElementById('loadingBox');
const errorMsg = document.getElementById('errorMsg');
const resultCard = document.getElementById('resultCard');
const metricCount = document.getElementById('metricCount');
const imageResultView = document.getElementById('imageResultView');
const videoResultView = document.getElementById('videoResultView');

function selectMode(mode) {
    state.currentMode = mode;
    const isImage = mode === 'image';

    document.getElementById('btnModeImage').classList.toggle('active', isImage);
    document.getElementById('btnModeVideo').classList.toggle('active', !isImage);

    fileInput.accept = isImage ? 'image/*' : 'video/*';
    uploadText.textContent = isImage ? 'Click or Drag & Drop a Crowd Image' : 'Click or Drag & Drop a Crowd Video';
    fileHint.textContent = isImage ? 'Supported formats: JPG, PNG, WEBP' : 'Supported formats: MP4, AVI, MOV, MKV, WEBM';
    state.selectedFile = null;
    state.currentResult = null;
    uploadPrompt.style.display = 'block';
    previewContainer.style.display = 'none';
    imagePreview.style.display = 'none';
    videoPreview.style.display = 'none';
    imagePreview.removeAttribute('src');
    videoPreview.removeAttribute('src');
    hideError();
    hideResult();
    btnAnalyze.disabled = true;
    fileInput.value = '';
}

function openFilePicker() {
    fileInput.click();
}

function onFileChosen(event) {
    const files = event.target.files || [];
    if (!files.length) return;

    const file = files[0];
    state.selectedFile = file;
    state.selectedType = state.currentMode === 'video' ? 'video' : 'image';
    updatePreview(file);
    btnAnalyze.disabled = false;
    hideError();
    hideResult();
}

function updatePreview(file) {
    const mimeTypeValid = file.type.startsWith('image/') || file.type.startsWith('video/') ||
        /\.(jpg|jpeg|png|webp|mp4|avi|mov|mkv|webm)$/i.test(file.name);

    if (!mimeTypeValid) {
        btnAnalyze.disabled = true;
        return;
    }

    uploadPrompt.style.display = 'none';
    previewContainer.style.display = 'block';
    if (file.type.startsWith('video/') || /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name)) {
        videoPreview.src = URL.createObjectURL(file);
        videoPreview.style.display = 'block';
        imagePreview.style.display = 'none';
    } else {
        imagePreview.src = URL.createObjectURL(file);
        imagePreview.style.display = 'block';
        videoPreview.style.display = 'none';
    }
    btnAnalyze.disabled = false;
}

function hideError() {
    errorMsg.style.display = 'none';
    errorMsg.textContent = '';
}

function showError(message) {
    errorMsg.textContent = message;
    errorMsg.style.display = 'block';
}

function hideResult() {
    resultCard.style.display = 'none';
    imageResultView.style.display = 'none';
    videoResultView.style.display = 'none';
    resultCard.classList.remove('visible');
}

async function submitAnalysis(event) {
    event.preventDefault();
    if (!state.selectedFile) {
        showError('Please choose a file first.');
        return;
    }

    hideError();
    btnAnalyze.disabled = true;
    loadingBox.style.display = 'block';

    const formData = new FormData();
    formData.append('file', state.selectedFile);

    try {
        const endpoint = state.currentMode === 'video' ? '/predict/video' : '/predict/image';
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.detail || data.message || data.error || 'Analysis failed.');
        }

        state.currentResult = data;
        renderResult(data);
    } catch (error) {
        console.error(error);
        showError(error.message || 'Unable to analyze the file.');
    } finally {
        loadingBox.style.display = 'none';
        btnAnalyze.disabled = false;
    }
}

function renderResult(data) {
    resultCard.style.display = 'block';
    metricCount.textContent = state.currentMode === 'video' ? (data.estimated_people ?? '--') : (data.predicted_count ?? '--');

    if (state.currentMode === 'video') {
        imageResultView.style.display = 'none';
        videoResultView.style.display = 'block';
        renderVideoResult(data);
        return;
    }

    videoResultView.style.display = 'none';
    imageResultView.style.display = 'block';
    renderImageResult(data);
}

function renderImageResult(data) {
    const densityMap = data.urls?.density_map;

    if (densityMap) {
        document.getElementById('resDensity').src = densityMap;
        document.getElementById('resDensitySolo').src = densityMap;
    }

    if (data.urls?.original_image) {
        document.getElementById('resOriginal').src = data.urls.original_image;
    }

    switchImageView('sideBySide');
}

function renderVideoResult(data) {
    const flow = data.direction_distribution || {};
    const directionMap = {
        right: Number(flow.right ?? flow.Right ?? 0),
        left: Number(flow.left ?? flow.Left ?? 0),
        forward: Number(flow.forward ?? flow.Forward ?? 0),
        backward: Number(flow.backward ?? flow.Backward ?? 0),
        stationary: Number(data.stationary_pct || 0),
        moving: Number(data.moving_pct || 0),
    };

    const directionalTotal = directionMap.right + directionMap.left + directionMap.forward + directionMap.backward;
    if (directionalTotal > 0 && directionMap.moving === 0) {
        directionMap.moving = 100;
        directionMap.stationary = 0;
    }

    document.getElementById('pctRight').textContent = `${directionMap.right}%`;
    document.getElementById('pctLeft').textContent = `${directionMap.left}%`;
    document.getElementById('pctForward').textContent = `${directionMap.forward}%`;
    document.getElementById('pctBackward').textContent = `${directionMap.backward}%`;
    document.getElementById('pctStationary').textContent = `${directionMap.stationary}%`;
    document.getElementById('pctMoving').textContent = `${directionMap.moving}%`;

    const dominant = data.dominant_direction || 'Mostly Stationary';
    document.getElementById('dominantDirectionText').textContent = dominant;

    const noMovement = Boolean(data.no_significant_movement);
    document.getElementById('noMovementMsg').style.display = noMovement ? 'block' : 'none';

}

function switchImageView(viewName) {
    const side = document.getElementById('viewSideBySide');
    const densityOnly = document.getElementById('viewDensityOnly');
    const tabs = document.querySelectorAll('.view-tab');

    const showSide = viewName === 'sideBySide';
    side.style.display = showSide ? 'block' : 'none';
    densityOnly.style.display = showSide ? 'none' : 'block';
    tabs.forEach((tab) => {
        const isActive = (tab.textContent.toLowerCase().includes('side') && showSide) || (tab.textContent.toLowerCase().includes('density') && !showSide);
        tab.classList.toggle('active', isActive);
    });
}

uploadBox.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadBox.classList.remove('dragover');
    const file = event.dataTransfer.files?.[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    onFileChosen({ target: fileInput });
});

selectMode('image');
