/**
 * directorio.js — Lógica del modal de registro biométrico
 * Soporta: subida de archivo de video Y grabación desde cámara en vivo.
 * Ambos modos envían el video al mismo endpoint POST /api/enrolar/
 */
(function () {
    // ── DOM ────────────────────────────────────────────────────
    const modal        = document.getElementById('modal-registro');
    const modalContent = document.getElementById('modal-content');
    const btnAbrir     = document.getElementById('btn-abrir-modal');
    const btnCerrar    = document.getElementById('btn-cerrar-modal');
    const form         = document.getElementById('form-enrolamiento');
    const btnSubmit    = document.getElementById('btn-submit');
    const submitIcon   = document.getElementById('submit-icon');
    const submitText   = document.getElementById('submit-text');
    const feedback     = document.getElementById('feedback');

    // Tabs
    const tabArchivo   = document.getElementById('tab-archivo');
    const tabCamara    = document.getElementById('tab-camara');
    const panelArchivo = document.getElementById('panel-archivo');
    const panelCamara  = document.getElementById('panel-camara');

    // File upload
    const inputVideo   = document.getElementById('input-video');
    const placeholder  = document.getElementById('upload-placeholder');
    const selected     = document.getElementById('upload-selected');
    const fileName     = document.getElementById('video-filename');
    const fileSize     = document.getElementById('video-filesize');

    // Camera
    const camPreview   = document.getElementById('cam-preview');
    const btnRec       = document.getElementById('btn-rec');
    const btnRecText   = document.getElementById('btn-rec-text');
    const btnStopRec   = document.getElementById('btn-stop-rec');
    const recIndicator = document.getElementById('rec-indicator');
    const recTimer     = document.getElementById('rec-timer');
    const camStatus    = document.getElementById('cam-status');

    let modoActivo = 'archivo'; // 'archivo' | 'camara'
    let mediaStream = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let recordedBlob = null;
    let timerInterval = null;
    let recSeconds = 0;

    // ── TABS ──────────────────────────────────────────────────
    const activeTabCls = 'bg-white dark:bg-zinc-700 shadow-sm text-brand-blue dark:text-brand-yellow';
    const inactiveTabCls = 'text-slate-500 dark:text-zinc-400 hover:text-slate-700';

    tabArchivo.addEventListener('click', () => {
        modoActivo = 'archivo';
        panelArchivo.classList.remove('hidden');
        panelCamara.classList.add('hidden');
        tabArchivo.className = `flex-1 py-2 px-4 rounded-lg text-sm font-semibold transition-all ${activeTabCls}`;
        tabCamara.className  = `flex-1 py-2 px-4 rounded-lg text-sm font-semibold transition-all ${inactiveTabCls}`;
        stopCamera();
        recordedBlob = null;
    });

    tabCamara.addEventListener('click', async () => {
        modoActivo = 'camara';
        panelCamara.classList.remove('hidden');
        panelArchivo.classList.add('hidden');
        tabCamara.className  = `flex-1 py-2 px-4 rounded-lg text-sm font-semibold transition-all ${activeTabCls}`;
        tabArchivo.className = `flex-1 py-2 px-4 rounded-lg text-sm font-semibold transition-all ${inactiveTabCls}`;
        recordedBlob = null;
        await initCamera();
    });

    // ── CAMERA ────────────────────────────────────────────────
    async function initCamera() {
        try {
            camStatus.textContent = 'Solicitando acceso a la cámara...';
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
                audio: false
            });
            camPreview.srcObject = mediaStream;
            camStatus.textContent = 'Cámara lista. Graba al menos 5 segundos mostrando tu rostro.';
        } catch (err) {
            camStatus.textContent = '❌ No se pudo acceder a la cámara: ' + err.message;
            camStatus.classList.add('text-red-500');
        }
    }

    function stopCamera() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(t => t.stop());
            mediaStream = null;
        }
        camPreview.srcObject = null;
        stopRecording();
    }

    // ── RECORDING ─────────────────────────────────────────────
    btnRec.addEventListener('click', () => {
        if (!mediaStream) return;
        recordedChunks = [];
        recordedBlob = null;
        recSeconds = 0;

        // Detect best supported MIME type
        const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
            ? 'video/webm;codecs=vp9'
            : 'video/webm';

        mediaRecorder = new MediaRecorder(mediaStream, { mimeType });

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(recordedChunks, { type: mimeType });
            camStatus.textContent = `✅ Grabación lista: ${(recordedBlob.size / (1024*1024)).toFixed(2)} MB — ${recSeconds}s`;
            camStatus.classList.remove('text-red-500');
            camStatus.classList.add('text-emerald-500');
        };

        mediaRecorder.start(1000); // chunks each second
        recIndicator.classList.remove('hidden');
        btnRec.disabled = true;
        btnRec.classList.add('opacity-50');
        btnStopRec.disabled = false;
        btnStopRec.classList.remove('opacity-50', 'bg-slate-300', 'dark:bg-zinc-700', 'text-slate-500');
        btnStopRec.classList.add('bg-zinc-800', 'text-white', 'hover:bg-zinc-900');

        timerInterval = setInterval(() => {
            recSeconds++;
            const m = String(Math.floor(recSeconds / 60)).padStart(2, '0');
            const s = String(recSeconds % 60).padStart(2, '0');
            recTimer.textContent = `${m}:${s}`;
        }, 1000);

        camStatus.textContent = 'Grabando... muestra tu rostro de frente.';
        camStatus.classList.remove('text-emerald-500');
    });

    btnStopRec.addEventListener('click', stopRecording);

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        recIndicator.classList.add('hidden');
        btnRec.disabled = false;
        btnRec.classList.remove('opacity-50');
        btnStopRec.disabled = true;
        btnStopRec.classList.add('opacity-50');
        clearInterval(timerInterval);
    }

    // ── MODAL OPEN/CLOSE ──────────────────────────────────────
    function abrirModal() {
        modal.classList.remove('hidden');
        modal.classList.add('flex');
        requestAnimationFrame(() => {
            modal.style.opacity = '1';
            modalContent.classList.remove('scale-95');
            modalContent.classList.add('scale-100');
        });
    }

    function cerrarModal() {
        modal.style.opacity = '0';
        modalContent.classList.remove('scale-100');
        modalContent.classList.add('scale-95');
        setTimeout(() => {
            modal.classList.add('hidden');
            modal.classList.remove('flex');
            stopCamera();
            resetForm();
        }, 300);
    }

    btnAbrir.addEventListener('click', abrirModal);
    btnCerrar.addEventListener('click', cerrarModal);
    modal.addEventListener('click', (e) => { if (e.target === modal) cerrarModal(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !modal.classList.contains('hidden')) cerrarModal(); });

    // ── FILE INPUT FEEDBACK ───────────────────────────────────
    inputVideo.addEventListener('change', () => {
        if (inputVideo.files.length > 0) {
            const file = inputVideo.files[0];
            placeholder.classList.add('hidden');
            selected.classList.remove('hidden');
            fileName.textContent = file.name;
            fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
        } else {
            placeholder.classList.remove('hidden');
            selected.classList.add('hidden');
        }
    });

    // ── FORM SUBMIT ───────────────────────────────────────────
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        feedback.classList.add('hidden');

        // Validate video source
        if (modoActivo === 'archivo' && (!inputVideo.files || inputVideo.files.length === 0)) {
            mostrarFeedback('error', '❌ Selecciona un archivo de video.');
            return;
        }
        if (modoActivo === 'camara' && !recordedBlob) {
            mostrarFeedback('error', '❌ Graba un video desde la cámara primero.');
            return;
        }

        // Loading state
        btnSubmit.disabled = true;
        submitIcon.innerHTML = '<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>';
        submitIcon.classList.add('animate-spin');
        submitText.textContent = 'Procesando biometría (esto puede tardar)...';

        const formData = new FormData();
        formData.append('carnet', document.getElementById('input-carnet').value);
        formData.append('nombre_completo', document.getElementById('input-nombre').value);

        if (modoActivo === 'archivo') {
            formData.append('video', inputVideo.files[0]);
        } else {
            formData.append('video', recordedBlob, 'grabacion_camara.webm');
        }

        const csrfToken = form.querySelector('[name=csrfmiddlewaretoken]').value;

        try {
            const response = await fetch('/api/enrolar/', {
                method: 'POST',
                headers: { 'X-CSRFToken': csrfToken },
                body: formData,
            });
            const data = await response.json();

            if (data.exito) {
                mostrarFeedback('success', `✅ ${data.mensaje}`);
                agregarFilaTabla(data.estudiante);
                setTimeout(cerrarModal, 2000);
            } else {
                const msg = data.mensaje || Object.values(data.errores || {}).join(' · ');
                mostrarFeedback('error', `❌ ${msg}`);
            }
        } catch (err) {
            mostrarFeedback('error', `❌ Error de conexión: ${err.message}`);
        } finally {
            btnSubmit.disabled = false;
            submitIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>';
            submitIcon.classList.remove('animate-spin');
            submitText.textContent = 'Registrar Estudiante';
        }
    });

    // ── HELPERS ────────────────────────────────────────────────
    function agregarFilaTabla(est) {
        const tbody = document.getElementById('tabla-estudiantes');
        const emptyRow = tbody.querySelector('td[colspan]');
        if (emptyRow) emptyRow.closest('tr').remove();

        const now = new Date().toLocaleDateString('es', { day:'2-digit', month:'short', year:'numeric', hour:'2-digit', minute:'2-digit' });
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-50 dark:hover:bg-zinc-800/50 transition-colors bg-emerald-50/50 dark:bg-emerald-900/10';
        tr.innerHTML = `
            <td class="px-6 py-4 font-mono font-semibold text-brand-blue dark:text-brand-yellow">${est.carnet}</td>
            <td class="px-6 py-4 font-medium">${est.nombre}</td>
            <td class="px-6 py-4"><span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>Activa</span></td>
            <td class="px-6 py-4 text-slate-500 dark:text-zinc-400">${now}</td>`;
        tbody.prepend(tr);

        const statTotal = document.getElementById('stat-total');
        statTotal.textContent = parseInt(statTotal.textContent || '0') + 1;
    }

    function mostrarFeedback(tipo, mensaje) {
        feedback.className = 'rounded-xl p-4 text-sm font-medium border';
        if (tipo === 'success') {
            feedback.classList.add('bg-emerald-50', 'text-emerald-700', 'border-emerald-200', 'dark:bg-emerald-900/20', 'dark:text-emerald-400', 'dark:border-emerald-800');
        } else {
            feedback.classList.add('bg-red-50', 'text-red-700', 'border-red-200', 'dark:bg-red-900/20', 'dark:text-red-400', 'dark:border-red-800');
        }
        feedback.textContent = mensaje;
    }

    function resetForm() {
        form.reset();
        feedback.className = 'hidden rounded-xl p-4 text-sm font-medium';
        placeholder.classList.remove('hidden');
        selected.classList.add('hidden');
        recordedBlob = null;
        recordedChunks = [];
        recSeconds = 0;
        camStatus.textContent = '';
        camStatus.classList.remove('text-emerald-500', 'text-red-500');
        btnSubmit.disabled = false;
        submitIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>';
        submitIcon.classList.remove('animate-spin');
        submitText.textContent = 'Registrar Estudiante';
        // Reset to file tab
        tabArchivo.click();
    }
})();
