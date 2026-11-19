const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const dropZoneTitle = document.getElementById('dropZoneTitle');
const dropZoneHint = document.getElementById('dropZoneHint');

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
const cancelIcon = document.getElementById('cancelIcon');
const cancelBtn = document.getElementById('cancelBtn');
const downloadBtn = document.getElementById('downloadBtn');
const downloadTranscriptBtn = document.getElementById('downloadTranscriptBtn');
const resetBtn = document.getElementById('resetBtn');
const debugLog = document.getElementById('debugLog');

const modeAnonymizeBtn = document.getElementById('modeAnonymizeBtn');
const modeTranscribeOnlyBtn = document.getElementById('modeTranscribeOnlyBtn');
const workerValue = document.getElementById('workerValue');
const workerSelect = document.getElementById('workerSelect');
const customWorkerWrap = document.getElementById('customWorkerWrap');
const customWorkerInput = document.getElementById('customWorkerInput');
const workerLimitText = document.getElementById('workerLimitText');
const transcribeToggle = document.getElementById('transcribeToggle');
const transcribeSwitchWrap = document.getElementById('transcribeSwitchWrap');
const transcribeStateBadge = document.getElementById('transcribeStateBadge');
const transcribeOptionsWrap = document.getElementById('transcribeOptionsWrap');
const asrModelSelect = document.getElementById('asrModelSelect');
const languageSelect = document.getElementById('languageSelect');
const computeDeviceSelect = document.getElementById('computeDeviceSelect');
const speakerCountSelect = document.getElementById('speakerCountSelect');
const diarizationSensitivitySelect = document.getElementById('diarizationSensitivitySelect');
const audioTracksSelect = document.getElementById('audioTracksSelect');
const customPromptInput = document.getElementById('customPromptInput');
const promptPresetsEl = document.getElementById('promptPresets');
const transcribeHintText = document.getElementById('transcribeHintText');

const recordingsSection = document.getElementById('recordingsSection');
const recordingsCount = document.getElementById('recordingsCount');
const recordingsList = document.getElementById('recordingsList');

const postTranscribeSection = document.getElementById('postTranscribeSection');
const togglePostTranscribeBtn = document.getElementById('togglePostTranscribeBtn');
const postTranscribeForm = document.getElementById('postTranscribeForm');
const postTranscribeTitle = document.getElementById('postTranscribeTitle');
const postAsrModelSelect = document.getElementById('postAsrModelSelect');
const postLanguageSelect = document.getElementById('postLanguageSelect');
const postComputeDeviceSelect = document.getElementById('postComputeDeviceSelect');
const postSpeakerCountSelect = document.getElementById('postSpeakerCountSelect');
const postDiarizationSensitivitySelect = document.getElementById('postDiarizationSensitivitySelect');
const postAudioTracksSelect = document.getElementById('postAudioTracksSelect');
const postCustomPromptInput = document.getElementById('postCustomPromptInput');
const postPromptPresetsEl = document.getElementById('postPromptPresets');
const postAudioSourceHint = document.getElementById('postAudioSourceHint');
const postTranscribeBtn = document.getElementById('postTranscribeBtn');

let currentTaskId = null;
let pollInterval = null;
let taskFinished = false;
let uploadAbortController = null;
let cancelRequested = false;
let lastTaskData = null;
let currentRunIsPostTranscribe = false;
let postHasOriginalAudio = true;

let selectedJobMode = localStorage.getItem('selectedJobMode') || 'anonymize';
if (selectedJobMode !== 'anonymize' && selectedJobMode !== 'transcribe_only') {
    selectedJobMode = 'anonymize';
}
let selectedWorkers = localStorage.getItem('selectedWorkers') || '4';
let customWorkers = localStorage.getItem('customWorkers') || '';
let serverCpuCount = null;
let serverMaxWorkers = null;
let serverCudaAvailable = null;
let serverAsrModelsMeta = new Map();
let promptPresetMap = {
    system_analyst:
        'Собеседование системного аналитика: REST, gRPC, HTTP 2.0, SOAP, JWT, Bearer, OAuth 2, PostgreSQL, Redis, MongoDB, Kafka, C4, Sequence, Activity, ERD, BPMN, Camunda, Swagger, Postman, Confluence, Jira, Git, Jenkins, Docs as Code',
    backend_devops:
        'Техническое собеседование: Python, Go, Java, Docker, Kubernetes, CI/CD, GitLab, PostgreSQL, Redis, Kafka, RabbitMQ, gRPC, REST API, GraphQL, OpenAPI, Prometheus, Grafana, Terraform, Linux',
    frontend_qa:
        'Техническое собеседование: JavaScript, TypeScript, React, Vue, Next.js, REST, GraphQL, WebSocket, Playwright, Cypress, Selenium, Postman, Swagger, Allure, Jira, Git',
};
let transcribeOnlyExtensions = [
    '.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v',
    '.mp3', '.wav', '.m4a', '.ogg', '.opus', '.flac', '.aac', '.wma',
];

const savedTranscribe = localStorage.getItem('transcribeEnabled');
let transcribeEnabled = savedTranscribe === null ? true : savedTranscribe !== 'false';
let selectedAsrModel = localStorage.getItem('selectedAsrModel') || 'gigaam-v3';
let selectedLanguage = localStorage.getItem('selectedLanguage') || 'ru';
let selectedComputeDevice = localStorage.getItem('selectedComputeDevice') || 'cpu';
let selectedSpeakers = localStorage.getItem('selectedSpeakers') || 'auto';
let selectedSensitivity = localStorage.getItem('selectedSensitivity') || 'normal';
let selectedAudioTracks = localStorage.getItem('selectedAudioTracks') || 'auto';
let selectedCustomPrompt = localStorage.getItem('selectedCustomPrompt') || '';

// Post-recording separate transcription form state (initialized from main settings)
let postAsrModel = selectedAsrModel;
let postLanguage = selectedLanguage;
let postComputeDevice = selectedComputeDevice;
let postSpeakers = selectedSpeakers;
let postSensitivity = selectedSensitivity;
let postAudioTracks = selectedAudioTracks;
let postCustomPrompt = selectedCustomPrompt;

function isCpuOnlyAsrModel(modelId) {
    if (serverAsrModelsMeta.has(modelId)) {
        return !serverAsrModelsMeta.get(modelId).supports_cuda;
    }
    return modelId === 'gigaam-v3' || modelId === 'parakeet';
}

function getGpuFallbackModel(currentModelId, selectEl) {
    const hasOpt = (id) => Boolean(selectEl && Array.from(selectEl.options).some((opt) => opt.value === id));
    if (currentModelId === 'gigaam-v3' && hasOpt('large-v3-ru')) {
        return 'large-v3-ru';
    }
    if (hasOpt('turbo')) return 'turbo';
    if (hasOpt('large-v3-ru')) return 'large-v3-ru';
    return 'turbo';
}

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
    if (!selectEl) return;
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
        } else if (!opt.disabled && opt.dataset.infoHint) {
            const metaEl = document.createElement('span');
            metaEl.className = 'clay-opt-meta';
            metaEl.textContent = opt.dataset.infoHint;
            itemBtn.appendChild(metaEl);
        }

        itemBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (opt.disabled) return;
            selectEl.value = opt.value;
            selectEl.dispatchEvent(new Event('change', { bubbles: true }));
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
                e.stopPropagation();
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

[
    workerSelect,
    asrModelSelect,
    languageSelect,
    computeDeviceSelect,
    speakerCountSelect,
    diarizationSensitivitySelect,
    audioTracksSelect,
    postAsrModelSelect,
    postLanguageSelect,
    postComputeDeviceSelect,
    postSpeakerCountSelect,
    postDiarizationSensitivitySelect,
    postAudioTracksSelect,
].forEach(enhanceSelect);

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

function applyComputeDeviceAvailabilityToPair(deviceSelectEl, modelId, currentDevice, onUpdate) {
    if (!deviceSelectEl) return { device: currentDevice, model: modelId };
    const cudaMissingOnHost = serverCudaAvailable === false;
    let nextDevice = currentDevice;
    let nextModel = modelId;

    if (nextDevice === 'cuda' && cudaMissingOnHost) {
        nextDevice = 'cpu';
    } else if (nextDevice === 'cuda' && isCpuOnlyAsrModel(nextModel)) {
        const targetSelect = deviceSelectEl === postComputeDeviceSelect ? postAsrModelSelect : asrModelSelect;
        nextModel = getGpuFallbackModel(nextModel, targetSelect);
    }

    const cpuOnlySelected = isCpuOnlyAsrModel(nextModel);
    Array.from(deviceSelectEl.options).forEach((opt) => {
        if (opt.value !== 'cuda') return;
        opt.disabled = cudaMissingOnHost;
        if (cudaMissingOnHost) {
            opt.dataset.limitHint = 'нет GPU';
            opt.dataset.infoHint = '';
            opt.title = 'NVIDIA CUDA не обнаружена (для Docker запустите с docker-compose.gpu.yml)';
        } else if (cpuOnlySelected) {
            opt.dataset.limitHint = '';
            opt.dataset.infoHint = nextModel === 'gigaam-v3' ? 'Whisper RU' : 'Whisper';
            opt.title = nextModel === 'gigaam-v3'
                ? 'При выборе GPU модель автоматически переключится на Whisper Large v3 RU'
                : 'При выборе GPU модель автоматически переключится на Whisper Turbo';
        } else {
            opt.dataset.limitHint = '';
            opt.dataset.infoHint = '';
            opt.title = '';
        }
    });

    if (onUpdate) onUpdate(nextDevice, nextModel);
    return { device: nextDevice, model: nextModel };
}

function applyComputeDeviceAvailability() {
    applyComputeDeviceAvailabilityToPair(
        computeDeviceSelect,
        selectedAsrModel,
        selectedComputeDevice,
        (nextDev, nextMod) => {
            selectedComputeDevice = nextDev;
            if (nextMod !== selectedAsrModel) {
                selectedAsrModel = nextMod;
                localStorage.setItem('selectedAsrModel', selectedAsrModel);
            }
        }
    );
}

function hasLatinTerms(text) {
    return /[A-Za-z]{2,}/.test(String(text || ''));
}

function updateTranscribeHint() {
    if (!transcribeHintText) return;
    if (selectedAsrModel === 'gigaam-v3') {
        if (selectedLanguage === 'en') {
            transcribeHintText.textContent =
                'GigaAM v3 специализирована на русском языке; для английской речи выберите Whisper.';
        } else if (hasLatinTerms(selectedCustomPrompt)) {
            transcribeHintText.textContent =
                'В словаре есть латинские термины — для нативного вывода латиницы выберите Whisper Large v3 RU или Turbo (для GigaAM v3 сработает автозамена по словарю).';
        } else {
            transcribeHintText.textContent =
                'Sber GigaAM v3: лучшая точность на русской речи (нарезка ≤20с с Silero VAD), пунктуация и регистр. Офлайн на CPU.';
        }
    } else if (selectedAsrModel === 'large-v3-ru') {
        transcribeHintText.textContent =
            'Русский файнтюн Whisper Large v3: высокая точность на русской речи и IT-терминах с поддержкой словаря (CPU/GPU).';
    } else {
        transcribeHintText.textContent =
            'Отдельный .txt с таймкодами и ролями. Словарь терминов передаётся напрямую в initial_prompt Whisper.';
    }
}

function renderModeControls() {
    const isTranscribeOnly = selectedJobMode === 'transcribe_only';
    if (modeAnonymizeBtn) {
        modeAnonymizeBtn.classList.toggle('is-active', !isTranscribeOnly);
        modeAnonymizeBtn.setAttribute('aria-selected', String(!isTranscribeOnly));
    }
    if (modeTranscribeOnlyBtn) {
        modeTranscribeOnlyBtn.classList.toggle('is-active', isTranscribeOnly);
        modeTranscribeOnlyBtn.setAttribute('aria-selected', String(isTranscribeOnly));
    }
    if (dropZoneTitle) {
        dropZoneTitle.textContent = isTranscribeOnly
            ? 'Перетащите видео или аудио сюда'
            : 'Перетащите видео сюда';
    }
    if (dropZoneHint) {
        dropZoneHint.textContent = isTranscribeOnly
            ? 'или выберите файл (.mp4, .mp3, .wav, .m4a, .mkv, .ogg...)'
            : 'или выберите файл (.mp4)';
    }
    if (fileInput) {
        fileInput.accept = isTranscribeOnly ? transcribeOnlyExtensions.join(',') : '.mp4';
    }
}

function renderTranscriptionControls() {
    const isTranscribeOnly = selectedJobMode === 'transcribe_only';
    const effectiveTranscribe = isTranscribeOnly || transcribeEnabled;

    transcribeToggle.checked = effectiveTranscribe;
    if (transcribeSwitchWrap) {
        transcribeSwitchWrap.classList.toggle('hidden', isTranscribeOnly);
    }
    transcribeOptionsWrap.classList.toggle('hidden', !effectiveTranscribe);
    if (isTranscribeOnly) {
        transcribeStateBadge.textContent = 'Режим транскрибации';
        transcribeStateBadge.classList.remove('off');
    } else {
        transcribeStateBadge.textContent = transcribeEnabled ? 'Вкл' : 'Выкл';
        transcribeStateBadge.classList.toggle('off', !transcribeEnabled);
    }

    applyComputeDeviceAvailability();

    if (Array.from(asrModelSelect.options).some((opt) => opt.value === selectedAsrModel)) {
        asrModelSelect.value = selectedAsrModel;
    } else if (asrModelSelect.options.length > 0) {
        selectedAsrModel = asrModelSelect.options[0].value;
        asrModelSelect.value = selectedAsrModel;
    }
    syncClaySelect(asrModelSelect);

    if (languageSelect) {
        if (Array.from(languageSelect.options).some((opt) => opt.value === selectedLanguage)) {
            languageSelect.value = selectedLanguage;
        } else if (languageSelect.options.length > 0) {
            selectedLanguage = languageSelect.options[0].value;
            languageSelect.value = selectedLanguage;
        }
        syncClaySelect(languageSelect);
    }

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

    if (diarizationSensitivitySelect) {
        if (Array.from(diarizationSensitivitySelect.options).some((opt) => opt.value === selectedSensitivity)) {
            diarizationSensitivitySelect.value = selectedSensitivity;
        } else {
            selectedSensitivity = 'normal';
            diarizationSensitivitySelect.value = 'normal';
        }
        syncClaySelect(diarizationSensitivitySelect);
    }

    if (audioTracksSelect) {
        if (Array.from(audioTracksSelect.options).some((opt) => opt.value === selectedAudioTracks)) {
            audioTracksSelect.value = selectedAudioTracks;
        } else {
            selectedAudioTracks = 'auto';
            audioTracksSelect.value = 'auto';
        }
        syncClaySelect(audioTracksSelect);
    }

    if (customPromptInput && customPromptInput.value !== selectedCustomPrompt) {
        customPromptInput.value = selectedCustomPrompt;
    }

    updateTranscribeHint();
}

function updatePostAudioSourceHint() {
    if (!postAudioSourceHint) return;
    const sourceText = postHasOriginalAudio
        ? 'Используется оригинальная аудиодорожка (без искажения голоса).'
        : 'Используется аудиодорожка из сохранённой записи.';
    if (postAsrModel === 'gigaam-v3' && postLanguage === 'en') {
        postAudioSourceHint.textContent =
            `${sourceText} GigaAM v3 специализирована на русском; для английской речи выберите Whisper.`;
    } else if (postAsrModel === 'gigaam-v3' && hasLatinTerms(postCustomPrompt)) {
        postAudioSourceHint.textContent =
            `${sourceText} Для латинских IT-терминов из словаря лучше подходит Whisper Large v3 RU / Turbo (для GigaAM v3 сработает автозамена).`;
    } else {
        postAudioSourceHint.textContent = sourceText;
    }
}

function renderPostTranscribeControls() {
    if (!postAsrModelSelect) return;
    applyComputeDeviceAvailabilityToPair(
        postComputeDeviceSelect,
        postAsrModel,
        postComputeDevice,
        (nextDev, nextMod) => {
            postComputeDevice = nextDev;
            postAsrModel = nextMod;
        }
    );

    if (Array.from(postAsrModelSelect.options).some((opt) => opt.value === postAsrModel)) {
        postAsrModelSelect.value = postAsrModel;
    } else if (postAsrModelSelect.options.length > 0) {
        postAsrModel = postAsrModelSelect.options[0].value;
        postAsrModelSelect.value = postAsrModel;
    }
    syncClaySelect(postAsrModelSelect);

    if (postLanguageSelect) {
        if (Array.from(postLanguageSelect.options).some((opt) => opt.value === postLanguage)) {
            postLanguageSelect.value = postLanguage;
        } else if (postLanguageSelect.options.length > 0) {
            postLanguage = postLanguageSelect.options[0].value;
            postLanguageSelect.value = postLanguage;
        }
        syncClaySelect(postLanguageSelect);
    }

    if (postComputeDeviceSelect) {
        if (Array.from(postComputeDeviceSelect.options).some((opt) => opt.value === postComputeDevice && !opt.disabled)) {
            postComputeDeviceSelect.value = postComputeDevice;
        } else {
            postComputeDevice = 'cpu';
            postComputeDeviceSelect.value = 'cpu';
        }
        syncClaySelect(postComputeDeviceSelect);
    }

    if (postSpeakerCountSelect) {
        if (Array.from(postSpeakerCountSelect.options).some((opt) => opt.value === postSpeakers)) {
            postSpeakerCountSelect.value = postSpeakers;
        } else {
            postSpeakers = 'auto';
            postSpeakerCountSelect.value = 'auto';
        }
        syncClaySelect(postSpeakerCountSelect);
    }

    if (postDiarizationSensitivitySelect) {
        if (Array.from(postDiarizationSensitivitySelect.options).some((opt) => opt.value === postSensitivity)) {
            postDiarizationSensitivitySelect.value = postSensitivity;
        } else {
            postSensitivity = 'normal';
            postDiarizationSensitivitySelect.value = 'normal';
        }
        syncClaySelect(postDiarizationSensitivitySelect);
    }

    if (postAudioTracksSelect) {
        if (Array.from(postAudioTracksSelect.options).some((opt) => opt.value === postAudioTracks)) {
            postAudioTracksSelect.value = postAudioTracks;
        } else {
            postAudioTracks = 'auto';
            postAudioTracksSelect.value = 'auto';
        }
        syncClaySelect(postAudioTracksSelect);
    }

    if (postCustomPromptInput && postCustomPromptInput.value !== postCustomPrompt) {
        postCustomPromptInput.value = postCustomPrompt;
    }

    updatePostAudioSourceHint();
}

function setJobMode(mode) {
    selectedJobMode = mode === 'transcribe_only' ? 'transcribe_only' : 'anonymize';
    localStorage.setItem('selectedJobMode', selectedJobMode);
    renderModeControls();
    renderTranscriptionControls();
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

function syncSelectOptionsIfChanged(selectEl, items) {
    if (!selectEl || !Array.isArray(items) || items.length === 0) return;
    const current = Array.from(selectEl.options);
    const unchanged =
        current.length === items.length &&
        current.every((opt, idx) => opt.value === items[idx].id && opt.textContent === items[idx].label);
    if (unchanged) return;

    selectEl.innerHTML = '';
    items.forEach((item) => {
        const opt = document.createElement('option');
        opt.value = item.id;
        opt.textContent = item.label;
        selectEl.appendChild(opt);
    });
}

function formatFileSize(bytes) {
    const n = Number(bytes) || 0;
    if (n <= 0) return '';
    if (n < 1024 * 1024) return `${Math.max(1, Math.round(n / 1024))} КБ`;
    return `${(n / (1024 * 1024)).toFixed(1)} МБ`;
}

function renderRecordingsList(recordings) {
    if (!recordingsSection || !recordingsList) return;
    if (!Array.isArray(recordings) || recordings.length === 0) {
        recordingsSection.classList.add('hidden');
        recordingsList.innerHTML = '';
        if (recordingsCount) recordingsCount.textContent = '0';
        return;
    }

    recordingsSection.classList.remove('hidden');
    if (recordingsCount) recordingsCount.textContent = String(recordings.length);
    recordingsList.innerHTML = '';

    recordings.forEach((rec) => {
        const item = document.createElement('div');
        item.className = 'recording-item';

        const top = document.createElement('div');
        top.className = 'recording-top';
        const nameEl = document.createElement('span');
        nameEl.className = 'recording-name';
        nameEl.textContent = rec.original_filename || `Запись ${String(rec.task_id).slice(0, 8)}`;
        top.appendChild(nameEl);

        const badge = document.createElement('span');
        badge.className = rec.transcript_ready ? 'recording-badge' : 'recording-badge muted';
        badge.textContent = rec.transcript_ready ? 'Транскрипт готов' : 'Без транскрипта';
        top.appendChild(badge);
        item.appendChild(top);

        const meta = document.createElement('div');
        meta.className = 'recording-meta';
        const metaParts = [];
        if (rec.created_at) {
            const dt = new Date(rec.created_at);
            if (!Number.isNaN(dt.getTime())) {
                metaParts.push(dt.toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }));
            }
        }
        if (rec.video_ready && rec.video_size_bytes) {
            metaParts.push(`MP4 ${formatFileSize(rec.video_size_bytes)}`);
        }
        if (rec.has_original_audio) {
            metaParts.push('ориг. голос в кэше');
        }
        meta.textContent = metaParts.join(' • ');
        item.appendChild(meta);

        const actions = document.createElement('div');
        actions.className = 'recording-actions';

        if (rec.video_ready) {
            const dlVid = document.createElement('button');
            dlVid.type = 'button';
            dlVid.className = 'recording-chip';
            dlVid.textContent = 'Скачать .mp4';
            dlVid.addEventListener('click', () => {
                window.location.href = `/api/download/${rec.task_id}`;
            });
            actions.appendChild(dlVid);
        }

        if (rec.transcript_ready) {
            const dlTxt = document.createElement('button');
            dlTxt.type = 'button';
            dlTxt.className = 'recording-chip';
            dlTxt.textContent = 'Скачать .txt';
            dlTxt.addEventListener('click', () => {
                window.location.href = `/api/download-transcript/${rec.task_id}`;
            });
            actions.appendChild(dlTxt);
        }

        if (rec.can_transcribe) {
            const trBtn = document.createElement('button');
            trBtn.type = 'button';
            trBtn.className = rec.transcript_ready ? 'recording-chip' : 'recording-chip recording-chip-primary';
            trBtn.textContent = rec.transcript_ready ? 'Пересоздать транскрипт' : 'Создать транскрипт';
            trBtn.title = `Запустить сразу (модель: ${selectedAsrModel}, язык: ${selectedLanguage})`;
            trBtn.addEventListener('click', () => {
                startPostTranscription(rec.task_id, {
                    asrModel: selectedAsrModel,
                    language: selectedLanguage,
                    computeDevice: selectedComputeDevice,
                    speakers: selectedSpeakers,
                    sensitivity: selectedSensitivity,
                    audioTracks: selectedAudioTracks,
                    customPrompt: selectedCustomPrompt,
                });
            });
            actions.appendChild(trBtn);

            const cfgBtn = document.createElement('button');
            cfgBtn.type = 'button';
            cfgBtn.className = 'recording-chip';
            cfgBtn.textContent = 'Настроить';
            cfgBtn.title = 'Выбрать модель, язык, словарь и число спикеров перед транскрибацией';
            cfgBtn.addEventListener('click', () => openRecordingPanel(rec));
            actions.appendChild(cfgBtn);
        }

        item.appendChild(actions);
        recordingsList.appendChild(item);
    });
}

function openRecordingPanel(rec) {
    if (!rec || !rec.task_id) return;
    stopPolling();
    closeAllClaySelects();
    currentTaskId = rec.task_id;
    taskFinished = true;
    cancelRequested = false;
    currentRunIsPostTranscribe = false;

    const taskData = {
        task_id: rec.task_id,
        status: 'completed',
        mode: rec.video_ready ? 'anonymize' : 'transcribe_only',
        video_ready: Boolean(rec.video_ready),
        transcript_ready: Boolean(rec.transcript_ready),
        has_original_audio: Boolean(rec.has_original_audio),
        can_transcribe: Boolean(rec.can_transcribe),
        asr_model: rec.asr_model || selectedAsrModel,
        language: rec.language || selectedLanguage,
        compute_device: selectedComputeDevice,
        speakers: selectedSpeakers,
        diarization_sensitivity: selectedSensitivity,
        audio_tracks: selectedAudioTracks,
        custom_prompt: selectedCustomPrompt,
        original_filename: rec.original_filename || '',
    };
    lastTaskData = taskData;

    uploadPanel.classList.add('hidden');
    statusPanel.classList.remove('hidden');
    spinner.classList.add('hidden');
    if (cancelIcon) cancelIcon.classList.add('hidden');
    if (cancelBtn) cancelBtn.classList.add('hidden');
    successIcon.classList.remove('hidden');
    debugLog.classList.add('hidden');
    debugLog.innerHTML = '';

    statusText.textContent = rec.original_filename || `Запись ${String(rec.task_id).slice(0, 8)}`;
    statusSubText.textContent = rec.transcript_ready
        ? 'Выберите модель и параметры ниже, чтобы пересоздать транскрипт.'
        : 'Выберите модель и параметры ниже, чтобы создать транскрипт.';

    bindDownloadButtons(taskData);
    renderPostTranscribeSection(taskData, { forceExpandForm: true });
    resetBtn.classList.remove('hidden');
}

async function loadRecordings() {
    if (!recordingsSection) return;
    try {
        const response = await fetch('/api/recordings');
        if (!response || !response.ok) return;
        const data = await response.json();
        renderRecordingsList(data.recordings || []);
    } catch (_err) {
        // Ignore gracefully if endpoint unavailable in mocked tests
    }
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
        if (Array.isArray(data.transcribe_only_extensions) && data.transcribe_only_extensions.length > 0) {
            transcribeOnlyExtensions = data.transcribe_only_extensions;
        }

        if (Array.isArray(data.asr_models) && data.asr_models.length > 0) {
            serverAsrModelsMeta = new Map();
            data.asr_models.forEach((m) => {
                serverAsrModelsMeta.set(m.id, {
                    supports_cuda: m.supports_cuda !== undefined ? Boolean(m.supports_cuda) : (m.id !== 'gigaam-v3' && m.id !== 'parakeet'),
                    fixed_language: m.fixed_language || null,
                });
            });
            syncSelectOptionsIfChanged(asrModelSelect, data.asr_models);
            syncSelectOptionsIfChanged(postAsrModelSelect, data.asr_models);
            if (!localStorage.getItem('selectedAsrModel') && data.default_asr_model) {
                selectedAsrModel = data.default_asr_model;
            }
        }

        if (Array.isArray(data.languages) && data.languages.length > 0) {
            syncSelectOptionsIfChanged(languageSelect, data.languages);
            syncSelectOptionsIfChanged(postLanguageSelect, data.languages);
            if (!localStorage.getItem('selectedLanguage') && data.default_language) {
                selectedLanguage = data.default_language;
            }
        }

        if (Array.isArray(data.diarization_sensitivities) && data.diarization_sensitivities.length > 0) {
            syncSelectOptionsIfChanged(diarizationSensitivitySelect, data.diarization_sensitivities);
            syncSelectOptionsIfChanged(postDiarizationSensitivitySelect, data.diarization_sensitivities);
            if (!localStorage.getItem('selectedSensitivity') && data.default_diarization_sensitivity) {
                selectedSensitivity = data.default_diarization_sensitivity;
            }
        }

        if (Array.isArray(data.audio_track_modes) && data.audio_track_modes.length > 0) {
            syncSelectOptionsIfChanged(audioTracksSelect, data.audio_track_modes);
            syncSelectOptionsIfChanged(postAudioTracksSelect, data.audio_track_modes);
            if (!localStorage.getItem('selectedAudioTracks') && data.default_audio_track_mode) {
                selectedAudioTracks = data.default_audio_track_mode;
            }
        }

        if (Array.isArray(data.prompt_presets) && data.prompt_presets.length > 0) {
            data.prompt_presets.forEach((p) => {
                if (p && p.id && typeof p.prompt === 'string') {
                    promptPresetMap[p.id] = p.prompt;
                }
            });
        }

        if (Array.isArray(data.compute_devices) && data.compute_devices.length > 0 && computeDeviceSelect) {
            syncSelectOptionsIfChanged(computeDeviceSelect, data.compute_devices);
            syncSelectOptionsIfChanged(postComputeDeviceSelect, data.compute_devices);
            if (!localStorage.getItem('selectedComputeDevice') && data.default_compute_device) {
                selectedComputeDevice = data.default_compute_device;
            }
        }

        if (savedTranscribe === null && localStorage.getItem('transcribeEnabled') === null && typeof data.transcription_default === 'boolean') {
            transcribeEnabled = data.transcription_default;
        }

        postAsrModel = selectedAsrModel;
        postLanguage = selectedLanguage;
        postComputeDevice = selectedComputeDevice;
        postSpeakers = selectedSpeakers;
        postSensitivity = selectedSensitivity;
        postAudioTracks = selectedAudioTracks;
        postCustomPrompt = selectedCustomPrompt;

        applyWorkerAvailability();
        renderModeControls();
        renderTranscriptionControls();
        renderPostTranscribeControls();
    } catch (error) {
        console.warn('Could not load server config:', error);
    }
}

function bindPromptPresetButtons(containerEl, onApply) {
    if (!containerEl || !containerEl.querySelectorAll) return;
    containerEl.querySelectorAll('.prompt-chip').forEach((btn) => {
        btn.addEventListener('click', () => {
            const presetId = (btn.dataset && btn.dataset.preset) || (btn.getAttribute && btn.getAttribute('data-preset'));
            if (presetId === 'clear') {
                onApply('');
            } else if (presetId && promptPresetMap[presetId]) {
                onApply(promptPresetMap[presetId]);
            }
        });
    });
}

if (modeAnonymizeBtn) {
    modeAnonymizeBtn.addEventListener('click', () => setJobMode('anonymize'));
}
if (modeTranscribeOnlyBtn) {
    modeTranscribeOnlyBtn.addEventListener('click', () => setJobMode('transcribe_only'));
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
    if (isCpuOnlyAsrModel(selectedAsrModel) && selectedComputeDevice === 'cuda') {
        selectedComputeDevice = 'cpu';
        localStorage.setItem('selectedComputeDevice', selectedComputeDevice);
    }
    renderTranscriptionControls();
});

if (languageSelect) {
    languageSelect.addEventListener('change', () => {
        selectedLanguage = languageSelect.value;
        localStorage.setItem('selectedLanguage', selectedLanguage);
        renderTranscriptionControls();
    });
}

if (computeDeviceSelect) {
    computeDeviceSelect.addEventListener('change', () => {
        selectedComputeDevice = computeDeviceSelect.value;
        localStorage.setItem('selectedComputeDevice', selectedComputeDevice);
        if (selectedComputeDevice === 'cuda' && isCpuOnlyAsrModel(selectedAsrModel)) {
            selectedAsrModel = getGpuFallbackModel(selectedAsrModel, asrModelSelect);
            localStorage.setItem('selectedAsrModel', selectedAsrModel);
        }
        renderTranscriptionControls();
    });
}

speakerCountSelect.addEventListener('change', () => {
    selectedSpeakers = speakerCountSelect.value;
    localStorage.setItem('selectedSpeakers', selectedSpeakers);
    syncClaySelect(speakerCountSelect);
});

if (diarizationSensitivitySelect) {
    diarizationSensitivitySelect.addEventListener('change', () => {
        selectedSensitivity = diarizationSensitivitySelect.value;
        localStorage.setItem('selectedSensitivity', selectedSensitivity);
        syncClaySelect(diarizationSensitivitySelect);
    });
}

if (audioTracksSelect) {
    audioTracksSelect.addEventListener('change', () => {
        selectedAudioTracks = audioTracksSelect.value;
        localStorage.setItem('selectedAudioTracks', selectedAudioTracks);
        syncClaySelect(audioTracksSelect);
    });
}

if (customPromptInput) {
    customPromptInput.addEventListener('input', () => {
        selectedCustomPrompt = customPromptInput.value.trim();
        localStorage.setItem('selectedCustomPrompt', selectedCustomPrompt);
        updateTranscribeHint();
    });
}

bindPromptPresetButtons(promptPresetsEl, (val) => {
    selectedCustomPrompt = val;
    localStorage.setItem('selectedCustomPrompt', selectedCustomPrompt);
    if (customPromptInput) customPromptInput.value = val;
    updateTranscribeHint();
});

// Post-recording separate transcription form events
if (postAsrModelSelect) {
    postAsrModelSelect.addEventListener('change', () => {
        postAsrModel = postAsrModelSelect.value;
        if (isCpuOnlyAsrModel(postAsrModel) && postComputeDevice === 'cuda') {
            postComputeDevice = 'cpu';
        }
        renderPostTranscribeControls();
    });
}

if (postLanguageSelect) {
    postLanguageSelect.addEventListener('change', () => {
        postLanguage = postLanguageSelect.value;
        syncClaySelect(postLanguageSelect);
        updatePostAudioSourceHint();
    });
}

if (postComputeDeviceSelect) {
    postComputeDeviceSelect.addEventListener('change', () => {
        postComputeDevice = postComputeDeviceSelect.value;
        if (postComputeDevice === 'cuda' && isCpuOnlyAsrModel(postAsrModel)) {
            postAsrModel = getGpuFallbackModel(postAsrModel, postAsrModelSelect);
        }
        renderPostTranscribeControls();
    });
}

if (postSpeakerCountSelect) {
    postSpeakerCountSelect.addEventListener('change', () => {
        postSpeakers = postSpeakerCountSelect.value;
        syncClaySelect(postSpeakerCountSelect);
    });
}

if (postDiarizationSensitivitySelect) {
    postDiarizationSensitivitySelect.addEventListener('change', () => {
        postSensitivity = postDiarizationSensitivitySelect.value;
        syncClaySelect(postDiarizationSensitivitySelect);
    });
}

if (postAudioTracksSelect) {
    postAudioTracksSelect.addEventListener('change', () => {
        postAudioTracks = postAudioTracksSelect.value;
        syncClaySelect(postAudioTracksSelect);
    });
}

if (postCustomPromptInput) {
    postCustomPromptInput.addEventListener('input', () => {
        postCustomPrompt = postCustomPromptInput.value.trim();
        updatePostAudioSourceHint();
    });
}

bindPromptPresetButtons(postPromptPresetsEl, (val) => {
    postCustomPrompt = val;
    if (postCustomPromptInput) postCustomPromptInput.value = val;
    updatePostAudioSourceHint();
});

if (togglePostTranscribeBtn) {
    togglePostTranscribeBtn.addEventListener('click', () => {
        if (!postTranscribeForm) return;
        const willShow = postTranscribeForm.classList.contains('hidden');
        postTranscribeForm.classList.toggle('hidden', !willShow);
        if (willShow) {
            renderPostTranscribeControls();
        }
    });
}

if (postTranscribeBtn) {
    postTranscribeBtn.addEventListener('click', () => {
        if (!currentTaskId) return;
        startPostTranscription(currentTaskId, {
            asrModel: postAsrModel,
            language: postLanguage,
            computeDevice: postComputeDevice,
            speakers: postSpeakers,
            sensitivity: postSensitivity,
            audioTracks: postAudioTracks,
            customPrompt: postCustomPrompt,
        });
    });
}

setSelectedWorkers(selectedWorkers);
renderModeControls();
renderTranscriptionControls();
renderPostTranscribeControls();
loadServerConfig();
loadRecordings();

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

function isAllowedFile(filename) {
    const lower = String(filename || '').toLowerCase();
    if (selectedJobMode === 'transcribe_only') {
        return transcribeOnlyExtensions.some((ext) => lower.endsWith(ext));
    }
    return lower.endsWith('.mp4');
}

function handleFiles(files) {
    const file = files[0];
    if (file && isAllowedFile(file.name)) {
        uploadFile(file);
    } else if (selectedJobMode === 'transcribe_only') {
        alert('Пожалуйста, выберите видео или аудио файл (.mp4, .mp3, .wav, .m4a, .mkv, .ogg...)');
    } else {
        alert('Пожалуйста, выберите файл .mp4');
    }
}

function prepareStatusPanelForRun(titleText, subText) {
    uploadPanel.classList.add('hidden');
    statusPanel.classList.remove('hidden');
    spinner.classList.remove('hidden');
    successIcon.classList.add('hidden');
    if (cancelIcon) cancelIcon.classList.add('hidden');
    downloadBtn.classList.add('hidden');
    downloadTranscriptBtn.classList.add('hidden');
    if (postTranscribeSection) postTranscribeSection.classList.add('hidden');
    if (postTranscribeForm) postTranscribeForm.classList.add('hidden');
    if (togglePostTranscribeBtn) togglePostTranscribeBtn.classList.add('hidden');
    resetBtn.classList.add('hidden');
    debugLog.classList.add('hidden');
    debugLog.innerHTML = '';
    if (cancelBtn) {
        cancelBtn.disabled = false;
        cancelBtn.textContent = 'Прервать';
        cancelBtn.classList.remove('hidden');
    }
    statusText.textContent = titleText;
    statusSubText.textContent = subText;
}

async function extractErrorDetail(response) {
    if (!response) return '';
    try {
        const body = await response.json();
        if (body && typeof body.detail === 'string' && body.detail.trim()) {
            return body.detail.trim();
        }
    } catch (_err) {
        // Ignore non-JSON error bodies
    }
    return '';
}

async function uploadFile(file) {
    stopPolling();
    taskFinished = false;
    cancelRequested = false;
    currentRunIsPostTranscribe = false;
    currentTaskId = null;
    lastTaskData = null;
    closeAllClaySelects();
    const workersForUpload = getWorkersForUpload();
    if (!workersForUpload) return;

    const isTranscribeOnly = selectedJobMode === 'transcribe_only';
    prepareStatusPanelForRun(
        'Загрузка...',
        isTranscribeOnly ? 'Отправка файла для транскрибации...' : 'Отправка видео на сервер...'
    );

    const formData = new FormData();
    formData.append('file', file);
    formData.append('workers', workersForUpload);
    formData.append('mode', selectedJobMode);
    formData.append('transcribe', String(isTranscribeOnly ? true : transcribeEnabled));
    formData.append('asr_model', selectedAsrModel);
    formData.append('language', selectedLanguage);
    formData.append('compute_device', selectedComputeDevice);
    formData.append('speakers', selectedSpeakers);
    formData.append('diarization_sensitivity', selectedSensitivity);
    formData.append('audio_tracks', selectedAudioTracks);
    formData.append('custom_prompt', selectedCustomPrompt);

    uploadAbortController = new AbortController();

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
            signal: uploadAbortController.signal,
        });
        uploadAbortController = null;

        if (!response.ok) {
            const detail = await extractErrorDetail(response);
            throw new Error(detail || 'Upload failed');
        }

        const data = await response.json();
        currentTaskId = data.task_id;

        if (cancelRequested) {
            fetch(`/api/cancel/${currentTaskId}`, { method: 'POST' }).catch(() => {});
            taskFinished = true;
            showCancelled('Обработка остановлена, временные файлы очищены.');
            return;
        }

        const time = new Date().toLocaleTimeString('ru-RU');
        const startNote = isTranscribeOnly
            ? 'Файл загружен. Старт транскрибации...'
            : 'Видео загружено. Старт обработки...';
        debugLog.innerHTML = `<p>[${time}] ${startNote}</p>`;
        debugLog.classList.remove('hidden');
        logCursor = 0;

        statusText.textContent = isTranscribeOnly ? 'Транскрибация...' : 'Обработка...';
        statusSubText.textContent = data.worker_count
            ? `Подготовка. Потоков: ${data.worker_count}.`
            : 'Подготовка к обработке.';
        checkStatus();
        stopPolling();
        pollInterval = setInterval(checkStatus, 1000);
    } catch (error) {
        uploadAbortController = null;
        if (error && (error.name === 'AbortError' || cancelRequested)) {
            taskFinished = true;
            showCancelled('Загрузка файла отменена.');
            return;
        }
        const detail = error && error.message && error.message !== 'Upload failed' ? ` (${error.message})` : '';
        showError(`Ошибка загрузки файла. Попробуйте снова.${detail}`);
    }
}

async function startPostTranscription(taskId, opts = {}) {
    stopPolling();
    taskFinished = false;
    cancelRequested = false;
    currentRunIsPostTranscribe = true;
    currentTaskId = taskId;
    closeAllClaySelects();
    const workersForRun = getWorkersForUpload() || '4';

    const chosenModel = opts.asrModel || selectedAsrModel;
    const chosenLang = opts.language || selectedLanguage;
    const chosenDev = opts.computeDevice || selectedComputeDevice;
    const chosenSpk = opts.speakers || selectedSpeakers;
    const chosenSens = opts.sensitivity || selectedSensitivity;
    const chosenTracks = opts.audioTracks || selectedAudioTracks;
    const chosenPrompt = opts.customPrompt !== undefined ? String(opts.customPrompt) : selectedCustomPrompt;

    // Remember user's latest transcription choices
    selectedAsrModel = chosenModel;
    selectedLanguage = chosenLang;
    selectedComputeDevice = chosenDev;
    selectedSpeakers = chosenSpk;
    selectedSensitivity = chosenSens;
    selectedAudioTracks = chosenTracks;
    selectedCustomPrompt = chosenPrompt;
    postAsrModel = chosenModel;
    postLanguage = chosenLang;
    postComputeDevice = chosenDev;
    postSpeakers = chosenSpk;
    postSensitivity = chosenSens;
    postAudioTracks = chosenTracks;
    postCustomPrompt = chosenPrompt;
    localStorage.setItem('selectedAsrModel', selectedAsrModel);
    localStorage.setItem('selectedLanguage', selectedLanguage);
    localStorage.setItem('selectedComputeDevice', selectedComputeDevice);
    localStorage.setItem('selectedSpeakers', selectedSpeakers);
    localStorage.setItem('selectedSensitivity', selectedSensitivity);
    localStorage.setItem('selectedAudioTracks', selectedAudioTracks);
    localStorage.setItem('selectedCustomPrompt', selectedCustomPrompt);

    prepareStatusPanelForRun('Транскрибация...', 'Запуск распознавания речи для готовой записи...');

    const formData = new FormData();
    formData.append('workers', workersForRun);
    formData.append('asr_model', chosenModel);
    formData.append('language', chosenLang);
    formData.append('compute_device', chosenDev);
    formData.append('speakers', chosenSpk);
    formData.append('diarization_sensitivity', chosenSens);
    formData.append('audio_tracks', chosenTracks);
    formData.append('custom_prompt', chosenPrompt);

    try {
        const response = await fetch(`/api/transcribe/${taskId}`, {
            method: 'POST',
            body: formData,
        });
        if (!response.ok) {
            const detail = await extractErrorDetail(response);
            throw new Error(detail || `HTTP ${response.status}`);
        }
        await response.json();

        const time = new Date().toLocaleTimeString('ru-RU');
        debugLog.innerHTML = `<p>[${time}] Старт отдельной транскрибации записи...</p>`;
        debugLog.classList.remove('hidden');
        logCursor = 0;

        checkStatus();
        stopPolling();
        pollInterval = setInterval(checkStatus, 1000);
    } catch (error) {
        console.error('Post-transcription start failed:', error);
        const detail = error && error.message ? ` (${error.message})` : '';
        showError(
            `Не удалось запустить транскрибацию для этой записи.${detail}`,
            lastTaskData ? { ...lastTaskData, is_post_transcribe: true } : null
        );
    }
}

let logCursor = 0;

async function checkStatus() {
    if (!currentTaskId || taskFinished || cancelRequested) return;

    try {
        const response = await fetch(`/api/status/${currentTaskId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        lastTaskData = data;
        if (typeof data.is_post_transcribe === 'boolean') {
            currentRunIsPostTranscribe = data.is_post_transcribe;
        }

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
        } else if (data.status === 'cancelled') {
            taskFinished = true;
            stopPolling();
            const isPost = Boolean(data.is_post_transcribe || currentRunIsPostTranscribe);
            const msg = isPost
                ? 'Транскрибация остановлена. Готовая запись сохранена.'
                : 'Обработка остановлена, временные файлы очищены.';
            showCancelled(msg, isPost ? { ...data, is_post_transcribe: true } : data);
        } else if (data.status === 'error') {
            taskFinished = true;
            stopPolling();
            showError('Ошибка при обработке файла.', data);
        } else if (data.status === 'processing') {
            statusText.textContent = (data.mode === 'transcribe_only' || data.is_post_transcribe)
                ? 'Транскрибация...'
                : 'Обработка...';
            statusSubText.textContent = data.message || 'Пожалуйста, подождите...';
        } else if (data.status === 'queued') {
            statusText.textContent = 'В очереди...';
            statusSubText.textContent = data.message || 'Ожидание очереди...';
        }
    } catch (error) {
        if (cancelRequested) return;
        console.error('Error polling status:', error);
        taskFinished = true;
        stopPolling();
        showError('Связь с сервером потеряна. Проверьте контейнер и попробуйте снова.');
    }
}

function renderPostTranscribeSection(data = {}, opts = {}) {
    if (!postTranscribeSection) return;
    const canTranscribe = Boolean(data.can_transcribe);
    if (!canTranscribe || !currentTaskId) {
        postTranscribeSection.classList.add('hidden');
        return;
    }

    const hasTranscript = Boolean(data.transcript_ready);
    postHasOriginalAudio = Boolean(data.has_original_audio);
    postAsrModel = data.asr_model || selectedAsrModel;
    postLanguage = data.language || selectedLanguage;
    postComputeDevice = data.compute_device || selectedComputeDevice;
    postSpeakers = data.speakers || selectedSpeakers;
    postSensitivity = data.diarization_sensitivity || selectedSensitivity;
    postAudioTracks = data.audio_tracks || selectedAudioTracks;
    postCustomPrompt = data.custom_prompt !== undefined ? String(data.custom_prompt) : selectedCustomPrompt;
    renderPostTranscribeControls();

    postTranscribeSection.classList.remove('hidden');

    if (hasTranscript) {
        if (togglePostTranscribeBtn) {
            togglePostTranscribeBtn.textContent = 'Пересоздать транскрипт (другая модель)';
            togglePostTranscribeBtn.classList.toggle('hidden', Boolean(opts.forceExpandForm));
        }
        if (postTranscribeTitle) postTranscribeTitle.textContent = 'Пересоздать транскрипт';
        if (postTranscribeBtn) postTranscribeBtn.textContent = 'Пересоздать транскрипт';
        if (postTranscribeForm) postTranscribeForm.classList.toggle('hidden', !opts.forceExpandForm);
    } else {
        if (togglePostTranscribeBtn) togglePostTranscribeBtn.classList.add('hidden');
        if (postTranscribeTitle) postTranscribeTitle.textContent = 'Создать транскрипт отдельно';
        if (postTranscribeBtn) postTranscribeBtn.textContent = 'Создать транскрипт';
        if (postTranscribeForm) postTranscribeForm.classList.remove('hidden');
    }
}

function bindDownloadButtons(data = {}) {
    const hasVideo = data.video_ready !== undefined
        ? Boolean(data.video_ready)
        : data.mode !== 'transcribe_only';
    const hasTranscript = Boolean(data.transcript_ready);

    downloadBtn.classList.toggle('hidden', !hasVideo);
    downloadTranscriptBtn.classList.toggle('hidden', !hasTranscript);

    if (hasVideo) {
        downloadBtn.onclick = () => {
            window.location.href = `/api/download/${currentTaskId}`;
        };
    }
    if (hasTranscript) {
        downloadTranscriptBtn.onclick = () => {
            window.location.href = `/api/download-transcript/${currentTaskId}`;
        };
    }
    return { hasVideo, hasTranscript };
}

function showSuccess(data = {}) {
    spinner.classList.add('hidden');
    if (cancelIcon) cancelIcon.classList.add('hidden');
    if (cancelBtn) cancelBtn.classList.add('hidden');
    successIcon.classList.remove('hidden');
    statusText.textContent = 'Готово!';

    const { hasVideo, hasTranscript } = bindDownloadButtons(data);
    if (hasVideo && hasTranscript) {
        statusSubText.textContent = 'Видео и транскрипт по ролям готовы.';
    } else if (hasVideo) {
        statusSubText.textContent = 'Видео успешно анонимизировано. При необходимости создайте транскрипт ниже.';
    } else {
        statusSubText.textContent = 'Транскрипт по ролям готов.';
    }

    renderPostTranscribeSection(data);
    resetBtn.classList.remove('hidden');

    const time = new Date().toLocaleTimeString('ru-RU');
    let readyNote = 'Готово к скачиванию!';
    if (hasVideo && hasTranscript) {
        readyNote = 'Видео (.mp4) и транскрипт (.txt) готовы к скачиванию!';
    } else if (hasVideo) {
        readyNote = 'Видео готово к скачиванию!';
    } else if (hasTranscript) {
        readyNote = 'Транскрипт (.txt) готов к скачиванию!';
    }
    debugLog.innerHTML += `<p style="color: #4ade80;">[${time}] ${readyNote}</p>`;
    debugLog.scrollTop = debugLog.scrollHeight;
}

function showCancelled(msg, data = null) {
    spinner.classList.add('hidden');
    successIcon.classList.add('hidden');
    if (cancelIcon) cancelIcon.classList.remove('hidden');
    if (cancelBtn) cancelBtn.classList.add('hidden');

    if (data && data.is_post_transcribe) {
        bindDownloadButtons(data);
        renderPostTranscribeSection(data);
    } else {
        downloadBtn.classList.add('hidden');
        downloadTranscriptBtn.classList.add('hidden');
        if (postTranscribeSection) postTranscribeSection.classList.add('hidden');
    }

    statusText.textContent = 'Прервано';
    statusSubText.textContent = msg;
    resetBtn.classList.remove('hidden');

    if (!debugLog.classList.contains('hidden')) {
        const time = new Date().toLocaleTimeString('ru-RU');
        debugLog.innerHTML += `<p style="color: #f59e0b;">[${time}] Обработка прервана пользователем.</p>`;
        debugLog.scrollTop = debugLog.scrollHeight;
    }
}

function showError(msg, data = null) {
    spinner.classList.add('hidden');
    if (cancelIcon) cancelIcon.classList.add('hidden');
    if (cancelBtn) cancelBtn.classList.add('hidden');
    if (data && data.is_post_transcribe) {
        bindDownloadButtons(data);
        renderPostTranscribeSection(data);
    }
    statusText.textContent = 'Ошибка';
    statusSubText.textContent = msg;
    resetBtn.classList.remove('hidden');
}

if (cancelBtn) {
    cancelBtn.onclick = async () => {
        if (taskFinished || cancelRequested) return;
        cancelRequested = true;
        cancelBtn.disabled = true;
        cancelBtn.textContent = 'Остановка...';

        if (uploadAbortController) {
            uploadAbortController.abort();
            return;
        }

        stopPolling();
        const taskIdToCancel = currentTaskId;
        let cancelPayload = null;
        if (taskIdToCancel) {
            try {
                const resp = await fetch(`/api/cancel/${taskIdToCancel}`, { method: 'POST' });
                if (resp && resp.ok) {
                    cancelPayload = await resp.json();
                }
            } catch (err) {
                console.warn('Cancel request error:', err);
            }
        }
        taskFinished = true;
        const mergedData = (lastTaskData || cancelPayload)
            ? { ...(lastTaskData || {}), ...(cancelPayload || {}) }
            : null;
        const isPost = Boolean(
            currentRunIsPostTranscribe || (mergedData && mergedData.is_post_transcribe)
        );
        if (isPost && mergedData) {
            mergedData.is_post_transcribe = true;
            lastTaskData = mergedData;
        }
        showCancelled(
            isPost
                ? 'Транскрибация остановлена. Готовая запись сохранена.'
                : 'Обработка остановлена, временные файлы очищены.',
            isPost ? mergedData : null
        );
    };
}

resetBtn.onclick = () => {
    stopPolling();
    closeAllClaySelects();
    currentTaskId = null;
    lastTaskData = null;
    taskFinished = false;
    cancelRequested = false;
    currentRunIsPostTranscribe = false;
    uploadAbortController = null;
    statusPanel.classList.add('hidden');
    uploadPanel.classList.remove('hidden');
    spinner.classList.remove('hidden');
    successIcon.classList.add('hidden');
    if (cancelIcon) cancelIcon.classList.add('hidden');
    if (cancelBtn) {
        cancelBtn.disabled = false;
        cancelBtn.textContent = 'Прервать';
        cancelBtn.classList.add('hidden');
    }
    downloadBtn.classList.add('hidden');
    downloadTranscriptBtn.classList.add('hidden');
    if (postTranscribeSection) postTranscribeSection.classList.add('hidden');
    if (postTranscribeForm) postTranscribeForm.classList.add('hidden');
    if (togglePostTranscribeBtn) togglePostTranscribeBtn.classList.add('hidden');
    resetBtn.classList.add('hidden');
    debugLog.classList.add('hidden');
    fileInput.value = '';
    renderWorkerControls();
    renderModeControls();
    renderTranscriptionControls();
    loadRecordings();
    if (serverCudaAvailable === null) {
        loadServerConfig();
    }
};

