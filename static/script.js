const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

// Theme Toggle Logic
const themeToggle = document.getElementById('themeToggle');
const sunIcon = document.querySelector('.sun-icon');
const moonIcon = document.querySelector('.moon-icon');

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    if (theme === 'dark') {
        sunIcon.classList.remove('hidden');
        moonIcon.classList.add('hidden');
    } else {
        sunIcon.classList.add('hidden');
        moonIcon.classList.remove('hidden');
    }
}

// Init theme
const savedTheme = localStorage.getItem('theme');
const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
if (savedTheme) {
    setTheme(savedTheme);
} else if (systemDark) {
    setTheme('dark');
} else {
    setTheme('light');
}

themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    setTheme(currentTheme === 'dark' ? 'light' : 'dark');
});
const uploadPanel = document.getElementById('uploadPanel');
const statusPanel = document.getElementById('statusPanel');
const statusText = document.getElementById('statusText');
const statusSubText = document.getElementById('statusSubText');
const spinner = document.getElementById('spinner');
const successIcon = document.getElementById('successIcon');
const downloadBtn = document.getElementById('downloadBtn');
const downloadTranscriptBtn = document.getElementById('downloadTranscriptBtn');
const resetBtn = document.getElementById('resetBtn');
const debugLog = document.getElementById('debugLog');
const workerValue = document.getElementById('workerValue');
const workerSelect = document.getElementById('workerSelect');
const customWorkerWrap = document.getElementById('customWorkerWrap');
const customWorkerInput = document.getElementById('customWorkerInput');
const workerLimitText = document.getElementById('workerLimitText');
const transcribeToggle = document.getElementById('transcribeToggle');
const transcribeStateBadge = document.getElementById('transcribeStateBadge');
const transcribeOptionsWrap = document.getElementById('transcribeOptionsWrap');
const asrModelSelect = document.getElementById('asrModelSelect');
const speakerCountSelect = document.getElementById('speakerCountSelect');

let currentTaskId = null;
let pollInterval = null;
let lastMessage = '';
let taskFinished = false;
let selectedWorkers = localStorage.getItem('selectedWorkers') || '4';
let customWorkers = localStorage.getItem('customWorkers') || '';
let serverCpuCount = null;
let serverMaxWorkers = null;

const savedTranscribe = localStorage.getItem('transcribeEnabled');
let transcribeEnabled = savedTranscribe === null ? true : savedTranscribe !== 'false';
let selectedAsrModel = localStorage.getItem('selectedAsrModel') || 'turbo';
let selectedSpeakers = localStorage.getItem('selectedSpeakers') || 'auto';

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

const WORKER_PRESETS = ['1', '2', '4', '6', '12', '16'];

function normalizeWorkerMode(value) {
    const stringValue = String(value || '4');
    if (WORKER_PRESETS.includes(stringValue) || stringValue === 'custom') return stringValue;
    if (/^\d+$/.test(stringValue)) {
        customWorkers = stringValue;
        return 'custom';
    }
    return '4';
}

function getWorkerLimitMessage() {
    const base = 'Количество обработчиков не может превышать (количество ядер * потоки) - 1, чтобы компьютер не выключился от перегрузки.';
    if (!serverMaxWorkers || !serverCpuCount) return base;
    return `Лимит этой системы: ${serverCpuCount} логических CPU - 1 = ${serverMaxWorkers}. ${base}`;
}

function getSelectedWorkerValue() {
    if (selectedWorkers === 'custom') return customWorkerInput.value.trim();
    return selectedWorkers;
}

function updateWorkerLimitMessage(errorText = '') {
    workerLimitText.textContent = errorText ? `${errorText} ${getWorkerLimitMessage()}` : getWorkerLimitMessage();
    workerLimitText.classList.toggle('error', Boolean(errorText));
}

function renderWorkerControls() {
    selectedWorkers = normalizeWorkerMode(selectedWorkers);
    workerSelect.value = selectedWorkers;
    customWorkerWrap.classList.toggle('hidden', selectedWorkers !== 'custom');
    if (selectedWorkers === 'custom') {
        customWorkerInput.value = customWorkers;
    }

    const value = getSelectedWorkerValue();
    workerValue.textContent = selectedWorkers === 'custom' ? `Другое${value ? ` (${value})` : ''}` : value;
    validateWorkerSelection();
}

function renderTranscriptionControls() {
    transcribeToggle.checked = transcribeEnabled;
    transcribeOptionsWrap.classList.toggle('hidden', !transcribeEnabled);
    transcribeStateBadge.textContent = transcribeEnabled ? 'Включена' : 'Выключена';
    transcribeStateBadge.classList.toggle('off', !transcribeEnabled);

    if (Array.from(asrModelSelect.options).some((opt) => opt.value === selectedAsrModel)) {
        asrModelSelect.value = selectedAsrModel;
    } else if (asrModelSelect.options.length > 0) {
        selectedAsrModel = asrModelSelect.options[0].value;
        asrModelSelect.value = selectedAsrModel;
    }

    if (Array.from(speakerCountSelect.options).some((opt) => opt.value === selectedSpeakers)) {
        speakerCountSelect.value = selectedSpeakers;
    } else {
        selectedSpeakers = 'auto';
        speakerCountSelect.value = 'auto';
    }
}

function validateWorkerSelection() {
    const rawValue = getSelectedWorkerValue();
    const workers = Number(rawValue);

    if (!Number.isInteger(workers) || workers < 1) {
        updateWorkerLimitMessage('Введите целое число воркеров от 1.');
        return false;
    }

    if (serverMaxWorkers && workers > serverMaxWorkers) {
        updateWorkerLimitMessage(`Выбрано ${workers}, максимум для этой системы: ${serverMaxWorkers}.`);
        return false;
    }

    updateWorkerLimitMessage();
    return true;
}

function getWorkersForUpload() {
    if (!validateWorkerSelection()) {
        if (selectedWorkers === 'custom') customWorkerInput.focus();
        return null;
    }
    return String(Number(getSelectedWorkerValue()));
}

function setSelectedWorkers(value) {
    selectedWorkers = normalizeWorkerMode(value);
    localStorage.setItem('selectedWorkers', selectedWorkers);
    renderWorkerControls();
}

function applyWorkerAvailability() {
    if (serverMaxWorkers) {
        customWorkerInput.max = String(serverMaxWorkers);
    }

    Array.from(workerSelect.options).forEach((option) => {
        if (option.value === 'custom') return;
        const workers = Number(option.value);
        const isUnavailable = Boolean(serverMaxWorkers && workers > serverMaxWorkers);
        option.disabled = isUnavailable;
        option.title = isUnavailable ? `Максимум для этой системы: ${serverMaxWorkers}` : '';
    });

    if (serverMaxWorkers && WORKER_PRESETS.includes(selectedWorkers) && Number(selectedWorkers) > serverMaxWorkers) {
        selectedWorkers = Number('4') <= serverMaxWorkers ? '4' : (WORKER_PRESETS.find((value) => Number(value) <= serverMaxWorkers) || 'custom');
    }

    renderWorkerControls();
}

async function loadServerConfig() {
    try {
        const response = await fetch('/api/config');
        if (!response.ok) return;
        const data = await response.json();
        serverCpuCount = Number(data.cpu_count) || null;
        serverMaxWorkers = Number(data.max_workers) || null;

        if (Array.isArray(data.asr_models) && data.asr_models.length > 0) {
            asrModelSelect.innerHTML = '';
            data.asr_models.forEach((model) => {
                const opt = document.createElement('option');
                opt.value = model.id;
                opt.textContent = model.label;
                asrModelSelect.appendChild(opt);
            });
            if (!localStorage.getItem('selectedAsrModel') && data.default_asr_model) {
                selectedAsrModel = data.default_asr_model;
            }
        }

        if (savedTranscribe === null && typeof data.transcription_default === 'boolean') {
            transcribeEnabled = data.transcription_default;
        }

        applyWorkerAvailability();
        renderTranscriptionControls();
    } catch (error) {
        console.warn('Could not load server config:', error);
    }
}

workerSelect.addEventListener('change', () => {
    setSelectedWorkers(workerSelect.value);
});

customWorkerInput.addEventListener('input', () => {
    customWorkers = customWorkerInput.value.trim();
    localStorage.setItem('customWorkers', customWorkers);
    renderWorkerControls();
});

transcribeToggle.addEventListener('change', () => {
    transcribeEnabled = transcribeToggle.checked;
    localStorage.setItem('transcribeEnabled', String(transcribeEnabled));
    renderTranscriptionControls();
});

asrModelSelect.addEventListener('change', () => {
    selectedAsrModel = asrModelSelect.value;
    localStorage.setItem('selectedAsrModel', selectedAsrModel);
});

speakerCountSelect.addEventListener('change', () => {
    selectedSpeakers = speakerCountSelect.value;
    localStorage.setItem('selectedSpeakers', selectedSpeakers);
});

setSelectedWorkers(selectedWorkers);
renderTranscriptionControls();
loadServerConfig();

// Drag and drop events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', handleDrop, false);
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', function() {
    if (this.files.length) handleFiles(this.files);
});

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFiles(files) {
    const file = files[0];
    if (file && file.name.toLowerCase().endsWith('.mp4')) {
        uploadFile(file);
    } else {
        alert('Пожалуйста, загрузите файл .mp4');
    }
}

async function uploadFile(file) {
    stopPolling();
    taskFinished = false;
    const workersForUpload = getWorkersForUpload();
    if (!workersForUpload) return;

    // UI Update
    uploadPanel.classList.add('hidden');
    statusPanel.classList.remove('hidden');
    statusText.textContent = 'Загрузка...';
    statusSubText.textContent = 'Пожалуйста, подождите, пока видео загружается.';
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('workers', workersForUpload);
    formData.append('transcribe', String(transcribeEnabled));
    formData.append('asr_model', selectedAsrModel);
    formData.append('speakers', selectedSpeakers);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const data = await response.json();
        currentTaskId = data.task_id;
        
        // Reset and show debug log
        const time = new Date().toLocaleTimeString('ru-RU');
        debugLog.innerHTML = `<p>[${time}] Видео загружено на сервер. Начало обработки...</p>`;
        debugLog.classList.remove('hidden');
        lastMessage = '';
        logCursor = 0;
        
        // Start polling
        statusText.textContent = 'Обработка...';
        statusSubText.textContent = 'Подготовка к анонимизации.';
        if (data.worker_count) {
            statusSubText.textContent += ` Воркеров: ${data.worker_count}.`;
        }
        checkStatus();
        stopPolling();
        pollInterval = setInterval(checkStatus, 1000);
        
    } catch (error) {
        showError('Ошибка загрузки. Попробуйте еще раз.');
    }
}

let logCursor = 0;

async function checkStatus() {
    if (!currentTaskId || taskFinished) return;
    
    try {
        const response = await fetch(`/api/status/${currentTaskId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        // Отрисовка всех непрочитанных логов
        if (data.logs && data.logs.length > logCursor) {
            const time = new Date().toLocaleTimeString('ru-RU');
            for (let i = logCursor; i < data.logs.length; i++) {
                debugLog.innerHTML += `<p>[${time}] ${data.logs[i]}</p>`;
            }
            debugLog.scrollTop = debugLog.scrollHeight;
            logCursor = data.logs.length;
        }

        if (data.status === 'completed') {
            if (taskFinished) return;
            taskFinished = true;
            stopPolling();
            showSuccess(data);
        } else if (data.status === 'error') {
            taskFinished = true;
            stopPolling();
            showError('Произошла ошибка при обработке.');
        } else if (data.status === 'processing') {
            statusText.textContent = 'Обработка...';
            statusSubText.textContent = data.message || 'Пожалуйста, подождите...';
        } else if (data.status === 'queued') {
            statusText.textContent = 'В очереди...';
            statusSubText.textContent = data.message || 'Ожидание своей очереди...';
        }
    } catch (error) {
        console.error('Error polling status:', error);
        taskFinished = true;
        stopPolling();
        showError('Связь с сервером потеряна (ошибка сети/сервера). Возможно, задача отменена или сервер перезапущен.');
    }
}

function showSuccess(data = {}) {
    spinner.classList.add('hidden');
    successIcon.classList.remove('hidden');
    statusText.textContent = 'Готово!';

    const hasTranscript = Boolean(data.transcript_ready);
    statusSubText.textContent = hasTranscript
        ? 'Видео анонимизировано, текстовая расшифровка по ролям готова.'
        : 'Ваше видео успешно анонимизировано.';

    downloadBtn.classList.remove('hidden');
    downloadTranscriptBtn.classList.toggle('hidden', !hasTranscript);
    resetBtn.classList.remove('hidden');
    
    const time = new Date().toLocaleTimeString('ru-RU');
    const readyNote = hasTranscript
        ? 'Видео и файл транскрибации (.txt) готовы к скачиванию!'
        : 'Видео готово и доступно к скачиванию!';
    debugLog.innerHTML += `<p style="color: #4ade80;">[${time}] ${readyNote}</p>`;
    debugLog.scrollTop = debugLog.scrollHeight;
    
    downloadBtn.onclick = () => {
        window.location.href = `/api/download/${currentTaskId}`;
    };

    if (hasTranscript) {
        downloadTranscriptBtn.onclick = () => {
            window.location.href = `/api/download-transcript/${currentTaskId}`;
        };
    }
}

function showError(msg) {
    spinner.classList.add('hidden');
    statusText.textContent = 'Ошибка';
    statusSubText.textContent = msg;
    resetBtn.classList.remove('hidden');
}

resetBtn.onclick = () => {
    // Reset UI
    stopPolling();
    currentTaskId = null;
    taskFinished = false;
    statusPanel.classList.add('hidden');
    uploadPanel.classList.remove('hidden');
    spinner.classList.remove('hidden');
    successIcon.classList.add('hidden');
    downloadBtn.classList.add('hidden');
    downloadTranscriptBtn.classList.add('hidden');
    resetBtn.classList.add('hidden');
    debugLog.classList.add('hidden');
    fileInput.value = '';
};
