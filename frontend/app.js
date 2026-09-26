/* ==========================================================================
   CAMELYON17 RESEARCH PLATFORM - JAVASCRIPT APP
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // --- STATE MANAGEMENT ---
    let selectedFile = null;
    let sessionHistory = [];
    const API_BASE = ''; // Same origin

    // --- DOM ELEMENTS ---
    const navItems = document.querySelectorAll('.nav-item');
    const viewPanels = document.querySelectorAll('.view-panel');
    const headerTitle = document.getElementById('header-page-title');
    const headerSub = document.getElementById('header-page-sub');

    // Upload Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const btnBrowse = document.getElementById('btn-browse');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const metaFilename = document.getElementById('meta-filename');
    const metaDims = document.getElementById('meta-dims');
    const metaSize = document.getElementById('meta-size');
    const resizeNotice = document.getElementById('resize-notice');
    const btnRemoveImage = document.getElementById('btn-remove-image');
    
    // Inference Mode Elements
    const inferenceModeSelect = document.getElementById('inference-mode');
    const singleModelGroup = document.getElementById('single-model-group');
    const singleModelSelect = document.getElementById('single-model-select');
    const btnAnalyze = document.getElementById('btn-analyze');

    // Results Elements
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const loadingState = document.getElementById('loading-state');
    const resultsContent = document.getElementById('results-content');
    const modelCardsContainer = document.getElementById('model-cards-container');
    const historyTbody = document.getElementById('history-tbody');
    const btnClearHistory = document.getElementById('btn-clear-history');

    // --- VIEW ROUTING META MAP ---
    const viewMeta = {
        'view-analysis': { title: 'Image Analysis', sub: 'Camelyon17 histopathology classification' },
        'view-history': { title: 'Analysis History', sub: 'Local session analysis log' },
        'view-model-comparison': { title: 'Model Comparison', sub: 'Compare training objectives & paradigms' },
        'view-model-performance': { title: 'Model Performance', sub: 'Final evaluation on unseen hospital node' },
        'view-research-overview': { title: 'Research Overview', sub: 'Domain-generalized FL research achievements' },
        'view-methodology': { title: 'Methodology', sub: 'Federated learning, differential privacy & domain generalization' },
        'view-dataset': { title: 'Dataset Specification', sub: 'Camelyon17 patch partitions & patient isolation' },
        'view-generalization': { title: 'Cross-Hospital Generalization', sub: 'Validation-to-test stability gap' },
        'view-privacy-analysis': { title: 'Privacy Analysis', sub: 'Differential privacy parameters & RDP bounds' },
        'view-computational-analysis': { title: 'Computational Overhead', sub: 'Training execution round times' },
        'view-literature-context': { title: 'Literature Benchmark Context', sub: 'Published WILDS leaderboard comparison' },
        'view-limitations': { title: 'Research Limitations', sub: 'Methodological scope & future work' },
        'view-configuration': { title: 'System Configuration', sub: 'Inference engine & checkpoint status' },
        'view-about': { title: 'About Project', sub: 'Study credentials & prototype notice' }
    };

    // --- NAVIGATION ROUTING ---
    function switchView(targetViewId) {
        if (!viewMeta[targetViewId]) targetViewId = 'view-analysis';

        navItems.forEach(item => {
            const viewId = item.getAttribute('data-view');
            if (viewId === targetViewId) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        viewPanels.forEach(panel => {
            if (panel.id === targetViewId) {
                panel.classList.add('active');
            } else {
                panel.classList.remove('active');
            }
        });

        // Update top header
        const meta = viewMeta[targetViewId];
        headerTitle.textContent = meta.title;
        headerSub.textContent = meta.sub;
    }

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const viewId = item.getAttribute('data-view');
            switchView(viewId);
            window.location.hash = item.getAttribute('href');
        });
    });

    // Handle initial hash load
    if (window.location.hash) {
        const hashView = 'view-' + window.location.hash.replace('#', '');
        if (viewMeta[hashView]) switchView(hashView);
    }

    // --- FILE UPLOAD HANDLERS ---
    btnBrowse.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files[0]) {
            handleFileSelect(dt.files[0]);
        }
    });

    function handleFileSelect(file) {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            alert('Unsupported file format. Please upload PNG, JPG, JPEG, or WEBP.');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            alert('File size exceeds 10 MB limit.');
            return;
        }

        selectedFile = file;
        metaFilename.textContent = file.name;
        metaSize.textContent = formatBytes(file.size);

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            
            // Determine image dimensions
            const img = new Image();
            img.onload = () => {
                const w = img.width;
                const h = img.height;
                metaDims.textContent = `${w} × ${h} px`;
                if (w !== 96 || h !== 96) {
                    resizeNotice.classList.remove('hidden');
                } else {
                    resizeNotice.classList.add('hidden');
                }
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);

        dropZone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        btnAnalyze.disabled = false;
    }

    btnRemoveImage.addEventListener('click', resetUpload);

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        btnAnalyze.disabled = true;
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // --- INFERENCE MODE SELECTOR ---
    inferenceModeSelect.addEventListener('change', () => {
        if (inferenceModeSelect.value === 'single') {
            singleModelGroup.classList.remove('hidden');
        } else {
            singleModelGroup.classList.add('hidden');
        }
    });

    // --- INFERENCE TRIGGER & API CALL ---
    btnAnalyze.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI state -> loading
        resultsPlaceholder.classList.add('hidden');
        resultsContent.classList.add('hidden');
        loadingState.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        const mode = inferenceModeSelect.value;
        try {
            if (mode === 'all') {
                const res = await fetch(`${API_BASE}/api/predict-all`, {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                renderAllResults(data);
            } else {
                const singleModel = singleModelSelect.value;
                formData.append('model_name', singleModel);
                const res = await fetch(`${API_BASE}/api/predict`, {
                    method: 'POST',
                    body: formData
                });
                const singleRes = await res.json();
                renderAllResults({ models: [singleRes] });
            }
        } catch (err) {
            console.error('Inference request failed:', err);
            alert('Failed to connect to inference service. Please check system status.');
        } finally {
            loadingState.classList.add('hidden');
            resultsContent.classList.remove('hidden');
        }
    });

    // --- RENDER MODEL RESULT CARDS ---
    function renderAllResults(data) {
        modelCardsContainer.innerHTML = '';
        const models = data.models || [];
        const timestamp = new Date().toLocaleTimeString();

        models.forEach(model => {
            const card = document.createElement('div');
            
            if (!model.available) {
                // Disabled / Coming soon card (DP-FedAvg)
                card.className = 'res-card disabled-card';
                card.innerHTML = `
                    <div class="res-card-header">
                        <div>
                            <div class="res-model-name">${model.name}</div>
                            <div class="res-model-type">${model.type}</div>
                        </div>
                        <span class="badge badge-disabled">COMING SOON</span>
                    </div>
                    <div class="res-prediction-box">
                        <div class="res-pred-label">Status</div>
                        <div class="res-pred-val text-muted">Checkpoint unavailable</div>
                    </div>
                    <div class="upload-hint" style="margin-top: 8px;">
                        Implementation attempted; reserved slot for future checkpoint update.
                    </div>
                `;
            } else {
                // Available Model Result Card
                const isBest = model.is_best || model.name === 'GroupDRO';
                card.className = `res-card ${isBest ? 'best-model-card' : ''}`;

                const badgeHtml = model.badge ? 
                    `<span class="badge ${isBest ? 'badge-success' : 'badge-primary'}">${model.badge}</span>` : '';

                const predClassLower = (model.prediction || '').toLowerCase();
                const normalProb = model.normal_percent || 0;
                const anomalousProb = model.anomalous_percent || 0;

                card.innerHTML = `
                    <div class="res-card-header">
                        <div>
                            <div class="res-model-name">${model.name}</div>
                            <div class="res-model-type">${model.type}</div>
                        </div>
                        ${badgeHtml}
                    </div>

                    <div class="res-prediction-box">
                        <div class="res-pred-label">Prediction</div>
                        <div class="res-pred-val ${predClassLower}">${model.prediction}</div>
                        <div class="res-confidence">Confidence: ${model.confidence_percent}%</div>
                    </div>

                    <div class="prob-bar-group">
                        <div class="prob-row">
                            <div class="prob-labels">
                                <span>Normal</span>
                                <span>${normalProb}%</span>
                            </div>
                            <div class="prob-track">
                                <div class="prob-fill normal" style="width: ${normalProb}%"></div>
                            </div>
                        </div>

                        <div class="prob-row">
                            <div class="prob-labels">
                                <span>Anomalous</span>
                                <span>${anomalousProb}%</span>
                            </div>
                            <div class="prob-track">
                                <div class="prob-fill anomalous" style="width: ${anomalousProb}%"></div>
                            </div>
                        </div>
                    </div>
                `;

                // Add to Session History
                addHistoryRecord({
                    time: timestamp,
                    filename: selectedFile ? selectedFile.name : 'image.png',
                    modelName: model.name,
                    prediction: model.prediction,
                    confidence: `${model.confidence_percent}%`
                });
            }

            modelCardsContainer.appendChild(card);
        });
    }

    // --- SESSION HISTORY HANDLERS ---
    function addHistoryRecord(record) {
        sessionHistory.unshift(record);
        renderHistoryTable();
    }

    function renderHistoryTable() {
        if (sessionHistory.length === 0) {
            historyTbody.innerHTML = `<tr><td colspan="5" class="text-center text-muted">No history recorded in this session yet.</td></tr>`;
            return;
        }

        historyTbody.innerHTML = sessionHistory.map(rec => `
            <tr>
                <td>${rec.time}</td>
                <td><span class="font-mono">${rec.filename}</span></td>
                <td><strong>${rec.modelName}</strong></td>
                <td><span class="badge ${rec.prediction === 'Normal' ? 'badge-success' : 'badge-warning'}">${rec.prediction}</span></td>
                <td class="font-mono">${rec.confidence}</td>
            </tr>
        `).join('');
    }

    btnClearHistory.addEventListener('click', () => {
        sessionHistory = [];
        renderHistoryTable();
    });

    // --- DYNAMIC MODEL COMPARISON CARDS IN MODEL COMPARISON TAB ---
    async function loadModelComparisonTab() {
        try {
            const res = await fetch(`${API_BASE}/api/models`);
            const data = await res.json();
            const container = document.getElementById('model-comparison-list');
            if (!container || !data.models) return;

            container.innerHTML = data.models.map(m => `
                <div class="card ${m.is_best ? 'border-primary' : ''}">
                    <div class="flex-between">
                        <h4>${m.title}</h4>
                        <span class="badge ${m.available ? (m.is_best ? 'badge-success' : 'badge-primary') : 'badge-disabled'}">
                            ${m.badge || (m.available ? 'AVAILABLE' : 'COMING SOON')}
                        </span>
                    </div>
                    <p class="text-secondary margin-top-sm" style="font-size: 13px;">${m.description}</p>
                    <div class="stats-table-grid margin-top-sm">
                        <div><span>Training Method:</span> <strong>${m.type}</strong></div>
                        <div><span>Checkpoint:</span> <span class="font-mono">${m.checkpoint}</span></div>
                        <div><span>Best Validation:</span> <strong>${m.best_round_epoch}</strong></div>
                        <div><span>Unseen-Test Accuracy:</span> <strong>${m.test_accuracy ? m.test_accuracy + '%' : 'N/A'}</strong></div>
                        <div><span>Balanced Accuracy:</span> <strong>${m.balanced_accuracy ? m.balanced_accuracy + '%' : 'N/A'}</strong></div>
                        <div><span>Privacy Guarantee:</span> <strong>${m.privacy}</strong></div>
                    </div>
                </div>
            `).join('');
        } catch (e) {
            console.error('Error loading model comparison tab:', e);
        }
    }

    loadModelComparisonTab();
});
