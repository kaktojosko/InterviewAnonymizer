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
const resetBtn = document.getElementById('resetBtn');
const debugLog = document.getElementById('debugLog');

let currentTaskId = null;
let pollInterval = null;
let lastMessage = '';

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
    // UI Update
    uploadPanel.classList.add('hidden');
    statusPanel.classList.remove('hidden');
    statusText.textContent = 'Загрузка...';
    statusSubText.textContent = 'Пожалуйста, подождите, пока видео загружается.';
    
    const formData = new FormData();
    formData.append('file', file);

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
        
        // Start polling
        statusText.textContent = 'Обработка...';
        statusSubText.textContent = 'Подготовка к анонимизации.';
        checkStatus();
        pollInterval = setInterval(checkStatus, 1000);
        
    } catch (error) {
        showError('Ошибка загрузки. Попробуйте еще раз.');
    }
}

async function checkStatus() {
    if (!currentTaskId) return;
    
    try {
        const response = await fetch(`/api/status/${currentTaskId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        if (data.status === 'completed') {
            clearInterval(pollInterval);
            showSuccess();
        } else if (data.status === 'error') {
            clearInterval(pollInterval);
            showError('Произошла ошибка при обработке.');
        } else if (data.status === 'processing') {
            statusText.textContent = 'Обработка...';
            statusSubText.textContent = data.message || 'Пожалуйста, подождите...';
            if (data.message && data.message !== lastMessage) {
                lastMessage = data.message;
                const time = new Date().toLocaleTimeString('ru-RU');
                debugLog.innerHTML += `<p>[${time}] ${data.message}</p>`;
                debugLog.scrollTop = debugLog.scrollHeight;
            }
        } else if (data.status === 'queued') {
            statusText.textContent = 'В очереди...';
            statusSubText.textContent = data.message || 'Ожидание своей очереди...';
            if (data.message && data.message !== lastMessage) {
                lastMessage = data.message;
                const time = new Date().toLocaleTimeString('ru-RU');
                debugLog.innerHTML += `<p>[${time}] ${data.message}</p>`;
                debugLog.scrollTop = debugLog.scrollHeight;
            }
        }
    } catch (error) {
        console.error('Error polling status:', error);
        clearInterval(pollInterval);
        showError('Связь с сервером потеряна (ошибка сети/сервера). Возможно, задача отменена или сервер перезапущен.');
    }
}

function showSuccess() {
    spinner.classList.add('hidden');
    successIcon.classList.remove('hidden');
    statusText.textContent = 'Готово!';
    statusSubText.textContent = 'Ваше видео успешно анонимизировано.';
    downloadBtn.classList.remove('hidden');
    resetBtn.classList.remove('hidden');
    
    const time = new Date().toLocaleTimeString('ru-RU');
    debugLog.innerHTML += `<p style="color: #4ade80;">[${time}] Видео готово и доступно к скачиванию!</p>`;
    debugLog.scrollTop = debugLog.scrollHeight;
    
    downloadBtn.onclick = () => {
        window.location.href = `/api/download/${currentTaskId}`;
    };
}

function showError(msg) {
    spinner.classList.add('hidden');
    statusText.textContent = 'Ошибка';
    statusSubText.textContent = msg;
    resetBtn.classList.remove('hidden');
}

resetBtn.onclick = () => {
    // Reset UI
    currentTaskId = null;
    statusPanel.classList.add('hidden');
    uploadPanel.classList.remove('hidden');
    spinner.classList.remove('hidden');
    successIcon.classList.add('hidden');
    downloadBtn.classList.add('hidden');
    resetBtn.classList.add('hidden');
    debugLog.classList.add('hidden');
    fileInput.value = '';
};
