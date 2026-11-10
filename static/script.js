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
const computeDeviceSelect = document.getElementById('computeDeviceSelect');
const speakerCountSelect = document.getElementById('speakerCountSelect');

let currentTaskId = null;
let pollInterval = null;
let taskFinished = false;
let selectedWorkers = localStorage.getItem('selectedWorkers') || '4';
let customWorkers = localStorage.getItem('customWorkers') || '';
let serverCpuCount = null;
let serverMaxWorkers = null;
let serverCudaAvailable = null;

const savedTranscribe = localStorage.getItem('transcribeEnabled');
let transcribeEnabled = savedTranscribe === null ? true : savedTranscribe !== 'false';
let selectedAsrModel = localStorage.getItem('selectedAsrModel') || 'turbo';
let selectedComputeDevice = localStorage.getItem('selectedComputeDevice') || 'cpu';
let selectedSpeakers = localStorage.getItem('selectedSpeakers') || 'auto';

// --- Custom Claymorphism Select Enhancement ---
const claySelectRegistry = new Map();

function closeAllClaySelects(exceptWrapper = null) {
    claySelectRegistry.forEach(({ wrapper, trigger }) => {
        if (wrapper !== exceptWrapper) {
            wrapper.classList.remove('is-open');
            trigger.setAttribute('aria-expanded', 'false');
        }
    });
}

function syncClaySelect(selectEl) {
    const entry = claySelectRegistry.get(selectEl);
    if (!entry) return;

    const { labelSpan, menu } = entry;
    const selectedOpt = selectEl.options[selectEl.selectedIndex] || selectEl.options[0];
    labelSpan.textContent = selectedOpt ? selectedOpt.textContent : '';

    menu.innerHTML = '';
    Array.from(selectEl.options).forEach((opt) => {
        const itemBtn = document.createElement('button');
        itemBtn.type = 'button';
        itemBtn.className = 'clay-select-option';
        itemBtn.setAttribute('role', 'option');
        const isSelected = opt.value === selectEl.value;
        itemBtn.setAttribute('aria-selected', String(isSelected));
        if (isSelected) itemBtn.classList.add('is-selected');

        if (opt.disabled) {
            itemBtn.disabled = true;
            itemBtn.classList.add('is-disabled');
            if (opt.title) itemBtn.title = opt.title;
        }

        const textEl = document.createElement('span');
        textEl.className = 'clay-opt-text';
        textEl.textContent = opt.textContent;
        itemBtn.appendChild(textEl);

        if (opt.disabled && opt.dataset.limitHint) {
            const metaEl = document.createElement('span');
            metaEl.className = 'clay-opt-meta';
            metaEl.textContent = opt.dataset.limitHint;
            itemBtn.appendChild(metaEl);
        } else if (isSelected) {
            const checkSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            checkSvg.setAttribute('class', 'clay-opt-check');
            checkSvg.setAttribute('width', '16');
            checkSvg.setAttribute('height', '16');
            checkSvg.setAttribute('viewBox', '0 0 24 24');
            checkSvg.setAttribute('fill', 'none');
            checkSvg.setAttribute('stroke', 'currentColor');
            checkSvg.setAttribute('stroke-width', '2.5');
            checkSvg.setAttribute('stroke-linecap', 'round');
            checkSvg.setAttribute('stroke-linejoin', 'round');
            const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
            poly.setAttribute('points', '20 6 9 17 4 12');
            checkSvg.appendChild(poly);
            itemBtn.appendChild(checkSvg);
        }

        itemBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (opt.disabled) return;
            if (selectEl.value !== opt.value) {
                selectEl.value = opt.value;
                selectEl.dispatchEvent(new Event('change', { bubbles: true }));
            }
            syncClaySelect(selectEl);
            closeAllClaySelects();
            entry.trigger.focus();
        });

        menu.appendChild(itemBtn);
    });
}

function enhanceSelect(selectEl) {
    if (!selectEl || claySelectRegistry.has(selectEl)) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'clay-select';
    selectEl.parentNode.insertBefore(wrapper, selectEl);
    wrapper.appendChild(selectEl);
    selectEl.classList.add('native-select-hidden');
    selectEl.tabIndex = -1;

    const trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'clay-select-trigger';
    trigger.setAttribute('aria-haspopup', 'listbox');
    trigger.setAttribute('aria-expanded', 'false');

    const labelSpan = document.createElement('span');
    labelSpan.className = 'clay-select-label';
    trigger.appendChild(labelSpan);

    const chevron = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    chevron.setAttribute('class', 'clay-select-chevron');
    chevron.setAttribute('width', '18');
    chevron.setAttribute('height', '18');
    chevron.setAttribute('viewBox', '0 0 24 24');
    chevron.setAttribute('fill', 'none');
    chevron.setAttribute('stroke', 'currentColor');
    chevron.setAttribute('stroke-width', '2.2');
    chevron.setAttribute('stroke-linecap', 'round');
    chevron.setAttribute('stroke-linejoin', 'round');
    const chevronPoly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    chevronPoly.setAttribute('points', '6 9 12 15 18 9');
    chevron.appendChild(chevronPoly);
    trigger.appendChild(chevron);

    const menu = document.createElement('div');
    menu.className = 'clay-select-menu';
    menu.setAttribute('role', 'listbox');

    wrapper.appendChild(trigger);
    wrapper.appendChild(menu);

    if (selectEl.id) {
        const assocLabel = document.querySelector(`label[for="${selectEl.id}"]`);
        if (assocLabel) {
            assocLabel.addEventListener('click', (e) => {
                e.preventDefault();
                trigger.focus();
                trigger.click();
            });
        }
    }

    trigger.addEventListener('click', (e) => {
        e.stopPropagation();
        const willOpen = !wrapper.classList.contains('is-open');
        closeAllClaySelects(wrapper);
        wrapper.classList.toggle('is-open', willOpen);
        trigger.setAttribute('aria-expanded', String(willOpen));
        if (willOpen) {
            const activeItem = menu.querySelector('.clay-select-option.is-selected:not(:disabled)') ||
                menu.querySelector('.clay-select-option:not(:disabled)');
            if (activeItem) activeItem.focus({ preventScroll: true });
        }
    });

    wrapper.addEventListener('keydown', (e) => {
        const enabledItems = Array.from(menu.querySelectorAll('.clay-select-option:not(:disabled)'));
        const isOpen = wrapper.classList.contains('is-open');

        if (e.key === 'Escape' && isOpen) {
            e.preventDefault();
            closeAllClaySelects();
            trigger.focus();
            return;
        }

        if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && enabledItems.length > 0) {
            e.preventDefault();
            if (!isOpen) {
                closeAllClaySelects(wrapper);
                wrapper.classList.add('is-open');
                trigger.setAttribute('aria-expanded', 'true');
                const selectedBtn = menu.querySelector('.clay-select-option.is-selected:not(:disabled)') || enabledItems[0];
                selectedBtn.focus();
                return;
            }
            const curIdx = enabledItems.indexOf(document.activeElement);
            const delta = e.key === 'ArrowDown' ? 1 : -1;
            const nextIdx = curIdx === -1 ? 0 : (curIdx + delta + enabledItems.length) % enabledItems.length;
            enabledItems[nextIdx].focus();
        }
    });

    claySelectRegistry.set(selectEl, { wrapper, trigger, labelSpan, menu });
    syncClaySelect(selectEl);
}

document.addEventListener('click', () => closeAllClaySelects());

[workerSelect, asrModelSelect, computeDeviceSelect, speakerCountSelect].forEach(enhanceSelect);

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
    if (!serverMaxWorkers || !serverCpuCount) {
        return 'Лимит: потоки CPU − 1 (защита от перегрузки). Максимум не всегда быстрее.';
    }
    return `Доступно до ${serverMaxWorkers} из ${serverCpuCount} потоков CPU (1 в резерве ОС). Максимум не всегда быстрее.`;
}

function getSelectedWorkerValue() {
    if (selectedWorkers === 'custom') return customWorkerInput.value.trim();
    return selectedWorkers;
}

function updateWorkerLimitMessage(errorText = '') {
    if (errorText) {
        const rangeHint = serverMaxWorkers ? `Допустимо от 1 до ${serverMaxWorkers}.` : '';
        workerLimitText.textContent = `${errorText} ${rangeHint}`.trim();
    } else {
        workerLimitText.textContent = getWorkerLimitMessage();
    }
    workerLimitText.classList.toggle('error', Boolean(errorText));
}

function renderWorkerControls() {
    selectedWorkers = normalizeWorkerMode(selectedWorkers);
    workerSelect.value = selectedWorkers;
    syncClaySelect(workerSelect);
    customWorkerWrap.classList.toggle('hidden', selectedWorkers !== 'custom');
    if (selectedWorkers === 'custom') {
        customWorkerInput.value = customWorkers;
    }

    const value = getSelectedWorkerValue();
    workerValue.textContent = selectedWorkers === 'custom' ? `Другое${value ? ` (${value})` : ''}` : value;
    validateWorkerSelection();
}

function applyComputeDeviceAvailability() {
    if (!computeDeviceSelect) return;
    Array.from(computeDeviceSelect.options).forEach((opt) => {
        if (opt.value !== 'cuda') return;
        const cudaUnsupportedByModel = selectedAsrModel === 'parakeet';
        const cudaMissingOnHost = serverCudaAvailable === false;
        const isDisabled = cudaMissingOnHost || cudaUnsupportedByModel;
        opt.disabled = isDisabled;
        if (cudaMissingOnHost) {
            opt.dataset.limitHint = 'нет GPU';
            opt.title = 'NVIDIA CUDA не обнаружена (для Docker запустите с docker-compose.gpu.yml)';
        } else if (cudaUnsupportedByModel) {
            opt.dataset.limitHint = 'только CPU';
            opt.title = 'Parakeet работает на CPU (INT8). Для GPU выберите Whisper Turbo или Small';
        } else {
            opt.dataset.limitHint = '';
            opt.title = '';
        }
    });

    const cudaOpt = Array.from(computeDeviceSelect.options).find((o) => o.value === 'cuda');
    if (selectedComputeDevice === 'cuda' && cudaOpt && cudaOpt.disabled) {
        selectedComputeDevice = 'cpu';
    }
}

function renderTranscriptionControls() {
    transcribeToggle.checked = transcribeEnabled;
    transcribeOptionsWrap.classList.toggle('hidden', !transcribeEnabled);
    transcribeStateBadge.textContent = transcribeEnabled ? 'Вкл' : 'Выкл';
    transcribeStateBadge.classList.toggle('off', !transcribeEnabled);

    if (Array.from(asrModelSelect.options).some((opt) => opt.value === selectedAsrModel)) {
        asrModelSelect.value = selectedAsrModel;
    } else if (asrModelSelect.options.length > 0) {
        selectedAsrModel = asrModelSelect.options[0].value;
        asrModelSelect.value = selectedAsrModel;
    }
    syncClaySelect(asrModelSelect);

    applyComputeDeviceAvailability();
    if (computeDeviceSelect) {
        if (Array.from(computeDeviceSelect.options).some((opt) => opt.value === selectedComputeDevice && !opt.disabled)) {
            computeDeviceSelect.value = selectedComputeDevice;
        } else {
            selectedComputeDevice = 'cpu';
            computeDeviceSelect.value = 'cpu';
        }
        syncClaySelect(computeDeviceSelect);
    }

    if (Array.from(speakerCountSelect.options).some((opt) => opt.value === selectedSpeakers)) {
        speakerCountSelect.value = selectedSpeakers;
    } else {
        selectedSpeakers = 'auto';
        speakerCountSelect.value = 'auto';
    }
    syncClaySelect(speakerCountSelect);
}

function validateWorkerSelection() {
    const rawValue = getSelectedWorkerValue();
    const workers = Number(rawValue);

    if (!Number.isInteger(workers) || workers < 1) {
        updateWorkerLimitMessage('Укажите целое число от 1.');
        return false;
    }

    if (serverMaxWorkers && workers > serverMaxWorkers) {
        updateWorkerLimitMessage(`Выбрано ${workers} — выше лимита системы.`);
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
        option.dataset.limitHint = isUnavailable ? `макс. ${serverMaxWorkers}` : '';
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
        if (typeof data.cuda_available === 'boolean') {
            serverCudaAvailable = data.cuda_available;
        }

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

        if (Array.isArray(data.compute_devices) && data.compute_devices.length > 0 && computeDeviceSelect) {
            computeDeviceSelect.innerHTML = '';
            data.compute_devices.forEach((dev) => {
                const opt = document.createElement('option');
                opt.value = dev.id;
                opt.textContent = dev.label;
                computeDeviceSelect.appendChild(opt);
            });
            if (!localStorage.getItem('selectedComputeDevice') && data.default_compute_device) {
                selectedComputeDevice = data.default_compute_device;
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
    renderTranscriptionControls();
});

if (computeDeviceSelect) {
    computeDeviceSelect.addEventListener('change', () => {
        selectedComputeDevice = computeDeviceSelect.value;
        localStorage.setItem('selectedComputeDevice', selectedComputeDevice);
        syncClaySelect(computeDeviceSelect);
    });
}

speakerCountSelect.addEventListener('change', () => {
    selectedSpeakers = speakerCountSelect.value;
    localStorage.setItem('selectedSpeakers', selectedSpeakers);
    syncClaySelect(speakerCountSelect);
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
        alert('Пожалуйста, выберите файл .mp4');
    }
}

async function uploadFile(file) {
    stopPolling();
    taskFinished = false;
    closeAllClaySelects();
    const workersForUpload = getWorkersForUpload();
    if (!workersForUpload) return;

    uploadPanel.classList.add('hidden');
    statusPanel.classList.remove('hidden');
    statusText.textContent = 'Загрузка...';
    statusSubText.textContent = 'Отправка видео на сервер...';
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('workers', workersForUpload);
    formData.append('transcribe', String(transcribeEnabled));
    formData.append('asr_model', selectedAsrModel);
    formData.append('compute_device', selectedComputeDevice);
    formData.append('speakers', selectedSpeakers);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const data = await response.json();
        currentTaskId = data.task_id;
        
        const time = new Date().toLocaleTimeString('ru-RU');
        debugLog.innerHTML = `<p>[${time}] Видео загружено. Старт обработки...</p>`;
        debugLog.classList.remove('hidden');
        logCursor = 0;
        
        statusText.textContent = 'Обработка...';
        statusSubText.textContent = data.worker_count
            ? `Подготовка. Потоков: ${data.worker_count}.`
            : 'Подготовка к обработке.';
        checkStatus();
        stopPolling();
        pollInterval = setInterval(checkStatus, 1000);
        
    } catch (error) {
        showError('Ошибка загрузки файла. Попробуйте снова.');
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
            showError('Ошибка при обработке видео.');
        } else if (data.status === 'processing') {
            statusText.textContent = 'Обработка...';
            statusSubText.textContent = data.message || 'Пожалуйста, подождите...';
        } else if (data.status === 'queued') {
            statusText.textContent = 'В очереди...';
            statusSubText.textContent = data.message || 'Ожидание очереди...';
        }
    } catch (error) {
        console.error('Error polling status:', error);
        taskFinished = true;
        stopPolling();
        showError('Связь с сервером потеряна. Проверьте контейнер и попробуйте снова.');
    }
}

function showSuccess(data = {}) {
    spinner.classList.add('hidden');
    successIcon.classList.remove('hidden');
    statusText.textContent = 'Готово!';

    const hasTranscript = Boolean(data.transcript_ready);
    statusSubText.textContent = hasTranscript
        ? 'Видео и транскрипт по ролям готовы.'
        : 'Видео успешно анонимизировано.';

    downloadBtn.classList.remove('hidden');
    downloadTranscriptBtn.classList.toggle('hidden', !hasTranscript);
    resetBtn.classList.remove('hidden');
    
    const time = new Date().toLocaleTimeString('ru-RU');
    const readyNote = hasTranscript
        ? 'Видео (.mp4) и транскрипт (.txt) готовы к скачиванию!'
        : 'Видео готово к скачиванию!';
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
