/* NemoAgent client UI — vanilla JS, talks to the local client process over /ui WebSocket. */
(() => {
  const $ = (id) => document.getElementById(id);
  const chat = $('chat'), input = $('input'), attBox = $('attachments');
  let ws = null, state = null, pending = [];        // pending attachments [{id,name,is_image,...}]
  let current = null;                                // current assistant bubble
  let reasoningCard = null, metrics = {};
  let sttBubble = null;

  /* ---------------------------------------------------------------- helpers */
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  function fmt(text) {
    let t = esc(text);
    t = t.replace(/```(\w*)\n([\s\S]*?)```/g, (_, l, c) => `<pre><code>${c}</code></pre>`);
    t = t.replace(/`([^`\n]+)`/g, '<code>$1</code>');
    t = t.replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
    return t;
  }
  function scroll() { chat.scrollTop = chat.scrollHeight; }
  function add(el) { chat.appendChild(el); scroll(); return el; }
  function div(cls, html) { const d = document.createElement('div'); d.className = cls; if (html !== undefined) d.innerHTML = html; return d; }
  function card(cls, title, body, open) {
    const d = document.createElement('details'); d.className = 'card ' + cls; if (open) d.open = true;
    d.innerHTML = `<summary>${title}</summary><pre>${esc(body)}</pre>`; return add(d);
  }
  function send(msg) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg)); }
  function setPill(id, cls, text) { const p = $(id); p.className = 'pill ' + cls; if (text) p.textContent = text; }
  function updateMetrics() {
    const parts = [];
    if (metrics.stt) parts.push(`STT ${metrics.stt} мс`);
    if (metrics.first_token) parts.push(`1-й токен ${metrics.first_token} мс`);
    if (metrics.first_audio) parts.push(`1-й звук ${metrics.first_audio} мс`);
    if (metrics.total) parts.push(`всего ${(metrics.total / 1000).toFixed(1)} с`);
    $('metrics').textContent = parts.join(' · ');
  }

  /* ---------------------------------------------------------------- websocket */
  function connect() {
    ws = new WebSocket(`ws://${location.host}/ui`);
    ws.onopen = () => { send({ type: 'get_state' }); };
    ws.onclose = () => { setPill('pill-server', 'err', 'клиент'); setTimeout(connect, 1500); };
    ws.onmessage = (e) => { try { handle(JSON.parse(e.data)); } catch (err) { console.error(err, e.data); } };
  }

  function handle(m) {
    switch (m.type) {
      case 'status': state = m; renderStatus(); break;
      case 'user_message': {
        const d = div('msg user');
        const src = m.source === 'voice' ? '🎙 голос' : '⌨ текст';
        let html = `<div class="src">${src}</div>${fmt(m.text || '')}`;
        if (m.attachments && m.attachments.length) html += `<div class="src">📎 ${m.attachments.length} влож.</div>`;
        d.innerHTML = html; add(d);
        current = null; reasoningCard = null; metrics = { stt: metrics.stt }; updateMetrics();
        break;
      }
      case 'round': if (m.round > 1) { current = null; } break;
      case 'delta': {
        if (!current) { current = add(div('msg assistant streaming')); current._text = ''; }
        current._text += m.content; current.innerHTML = fmt(current._text); scroll();
        break;
      }
      case 'reasoning': {
        if (!reasoningCard) { reasoningCard = card('reasoning', '🧠 рассуждения', '', false); }
        const pre = reasoningCard.querySelector('pre'); pre.textContent += m.content; break;
      }
      case 'tool_call': card('tool', `🔧 <b>${esc(m.name)}</b>`, JSON.stringify(m.arguments ?? m.raw, null, 1), false); current = null; break;
      case 'tool_result': {
        const r = m.result || {}; const ok = !r.error;
        card('tool', `${ok ? '✅' : '⚠️'} <b>${esc(m.name)}</b> · ${m.ms} мс`, JSON.stringify(r, null, 1).slice(0, 4000), !ok); break;
      }
      case 'client_tool_start': $('stt-state').textContent = `выполняю: ${m.summary.slice(0, 80)}`; break;
      case 'client_tool_done': $('stt-state').textContent = m.ok ? '' : `⚠ ${m.name} завершился с ошибкой`; break;
      case 'memory': card('memory', `🗂 память: ${m.items.length} совпад.`, m.items.map((i) => `[${i.kind} ${i.score}] ${i.text}`).join('\n\n'), false); break;
      case 'wait': $('stt-state').textContent = m.stage === 'retry' ? `сервер NVIDIA перегружен, повтор ${m.attempt}…` : 'жду модель…'; break;
      case 'notice': add(div('notice', esc(m.message))); break;
      case 'error': add(div('errline', '⚠ ' + esc(m.message))); break;
      case 'done': {
        if (current) current.classList.remove('streaming');
        if (m.first_token_ms) metrics.first_token = m.first_token_ms;
        if (m.total_ms) metrics.total = m.total_ms; updateMetrics();
        if (m.finish_reason === 'interrupted') add(div('notice', 'прервано'));
        $('stt-state').textContent = ''; current = null; break;
      }
      case 'tts_first_audio': metrics.first_audio = m.ms; updateMetrics(); break;
      case 'interrupted': if (current) current.classList.remove('streaming'); break;
      case 'cleared': chat.innerHTML = ''; current = null; reasoningCard = null; break;
      case 'mic': { const el = $('mic-level'); el.style.width = Math.round(m.level * 100) + '%'; el.classList.toggle('speech', !!m.speech); break; }
      case 'stt': {
        const s = $('stt-state');
        if (m.state === 'transcribing') s.textContent = `распознаю ${m.duration} с…`;
        else if (m.state === 'done') { s.textContent = ''; metrics.stt = m.ms; }
        else if (m.state === 'empty') s.textContent = 'ничего не распознано';
        else if (m.state === 'echo') s.textContent = 'эхо собственной речи — пропущено';
        else if (m.state === 'error') s.textContent = 'ошибка STT: ' + m.message;
        break;
      }
      case 'confirm': showConfirm(m); break;
      case 'confirm_expired': hideConfirm(); break;
    }
  }

  /* ---------------------------------------------------------------- status & settings */
  function renderStatus() {
    setPill('pill-server', state.server ? 'ok' : 'err', state.server ? 'сервер' : 'нет сервера');
    const st = state.stt || '', tt = state.tts || '';
    setPill('pill-stt', st.startsWith('ready') ? 'ok' : st.startsWith('error') ? 'err' : st === 'off' ? '' : 'warn', 'STT ' + st.replace('ready ', ''));
    setPill('pill-tts', tt.startsWith('ready') ? 'ok' : tt.startsWith('error') ? 'err' : tt === 'off' ? '' : 'warn', 'TTS ' + tt.replace('ready ', ''));
    const si = state.server_info || {};
    setPill('pill-vision', si.vision ? 'ok' : 'warn', si.vision ? 'vision' : 'vision off');
    const mem = si.memory ? Object.values(si.memory).reduce((a, b) => a + b, 0) : 0;
    setPill('pill-memory', 'ok', `память ${mem}`);
    $('model').textContent = si.model ? '· ' + si.model.split('/').pop() : '';
    $('btn-listen').classList.toggle('active', !!state.listening);
    $('pill-stt').title = state.mic ? 'микрофон: ' + state.mic : '';
    const s = state.settings || {};
    for (const k of ['tts_mode', 'confirm', 'stt_language']) $('s-' + k).value = s[k];
    for (const k of ['auto_listen', 'barge_in', 'tools_enabled']) $('s-' + k).checked = !!s[k];
    fillVoices('s-voice_ru', state.voices.ru, s.voice_ru); fillVoices('s-voice_en', state.voices.en, s.voice_en);
    $('s-tts_speed').value = s.tts_speed; $('s-tts_speed-v').textContent = Number(s.tts_speed).toFixed(2);
    $('settings-status').textContent = `сессия ${state.session_id || '—'} · mic ${state.mic || ''}`;
  }
  function fillVoices(id, list, val) {
    const sel = $(id); if (sel.options.length !== list.length) { sel.innerHTML = list.map((v) => `<option value="${v}">${v}</option>`).join(''); }
    sel.value = val;
  }
  function pushSettings(patch) { send({ type: 'settings', settings: patch }); }
  $('btn-settings').onclick = () => $('settings').classList.toggle('hidden');
  for (const k of ['tts_mode', 'confirm', 'stt_language', 'voice_ru', 'voice_en']) $('s-' + k).onchange = (e) => pushSettings({ [k]: e.target.value });
  for (const k of ['auto_listen', 'barge_in', 'tools_enabled']) $('s-' + k).onchange = (e) => pushSettings({ [k]: e.target.checked });
  $('s-tts_speed').oninput = (e) => { $('s-tts_speed-v').textContent = Number(e.target.value).toFixed(2); };
  $('s-tts_speed').onchange = (e) => pushSettings({ tts_speed: Number(e.target.value) });
  $('btn-say').onclick = () => send({ type: 'say', text: 'Привет! Голос работает. Hello, the voice is working.' });

  /* ---------------------------------------------------------------- composer */
  function renderAttachments() {
    attBox.innerHTML = '';
    for (const a of pending) {
      const c = div('chip' + (a.pending ? ' pending' : ''));
      c.innerHTML = `${a.is_image && a.preview ? `<img src="${a.preview}">` : '📄'} <span>${esc(a.name)}</span> <span class="x" title="убрать">✕</span>`;
      c.querySelector('.x').onclick = () => { pending = pending.filter((p) => p !== a); renderAttachments(); };
      attBox.appendChild(c);
    }
  }
  async function uploadFiles(files) {
    for (const f of files) {
      const entry = { name: f.name, is_image: f.type.startsWith('image/'), pending: true };
      if (entry.is_image) entry.preview = URL.createObjectURL(f);
      pending.push(entry); renderAttachments();
      const fd = new FormData(); fd.append('file', f, f.name);
      try {
        const r = await fetch('/ui/upload', { method: 'POST', body: fd }); const j = await r.json();
        if (j.error) { add(div('errline', '⚠ ' + esc(j.error))); pending = pending.filter((p) => p !== entry); }
        else { Object.assign(entry, j, { pending: false }); }
      } catch (e) { add(div('errline', '⚠ загрузка не удалась: ' + esc(e.message))); pending = pending.filter((p) => p !== entry); }
      renderAttachments();
    }
  }
  function sendMessage() {
    const text = input.value.trim();
    if (pending.some((p) => p.pending)) { $('stt-state').textContent = 'дождитесь загрузки вложений…'; return; }
    const ids = pending.map((p) => p.id).filter(Boolean);
    if (!text && !ids.length) return;
    send({ type: 'send', text, attachments: ids });
    input.value = ''; input.style.height = 'auto'; pending = []; renderAttachments();
  }
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
  input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = Math.min(200, input.scrollHeight) + 'px'; });
  input.addEventListener('paste', (e) => {
    const files = [...(e.clipboardData?.items || [])].filter((i) => i.kind === 'file').map((i) => i.getAsFile()).filter(Boolean);
    if (files.length) { e.preventDefault(); uploadFiles(files.map((f, i) => f.name ? f : new File([f], `paste_${Date.now()}_${i}.png`, { type: f.type }))); }
  });
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', (e) => { e.preventDefault(); if (e.dataTransfer?.files?.length) uploadFiles([...e.dataTransfer.files]); });
  $('btn-send').onclick = sendMessage;
  $('btn-attach').onclick = () => $('file-input').click();
  $('file-input').onchange = (e) => { uploadFiles([...e.target.files]); e.target.value = ''; };
  $('btn-shot').onclick = () => {
    const entry = { name: 'скриншот…', is_image: true, pending: true }; pending.push(entry); renderAttachments();
    const rid = String(Date.now());
    const h = (e) => { const m = JSON.parse(e.data); if (m.type === 'attachment' && m.request_id === rid) { ws.removeEventListener('message', h);
      if (m.error) { add(div('errline', '⚠ ' + esc(m.error))); pending = pending.filter((p) => p !== entry); } else Object.assign(entry, m, { pending: false, name: m.name }); renderAttachments(); } };
    ws.addEventListener('message', h); send({ type: 'screenshot', request_id: rid });
  };
  $('btn-stop').onclick = () => send({ type: 'interrupt' });
  $('btn-new').onclick = () => send({ type: 'new_session' });
  $('btn-listen').onclick = () => pushSettings({ auto_listen: !(state?.settings?.auto_listen) });

  /* push-to-talk: hold the button (mouse/touch) or hold Space when the input is not focused */
  const ptt = $('btn-ptt'); let held = false;
  const down = (e) => { e.preventDefault(); if (held) return; held = true; ptt.classList.add('rec'); send({ type: 'ptt', state: 'down' }); };
  const up = () => { if (!held) return; held = false; ptt.classList.remove('rec'); send({ type: 'ptt', state: 'up' }); };
  ptt.addEventListener('mousedown', down); ptt.addEventListener('touchstart', down, { passive: false });
  window.addEventListener('mouseup', up); window.addEventListener('touchend', up);
  const typing = () => ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
  window.addEventListener('keydown', (e) => { if (e.code === 'Space' && !typing() && !e.repeat && !e.ctrlKey && !e.altKey && !e.metaKey) down(e); });
  window.addEventListener('keyup', (e) => { if (e.code === 'Space' && held) up(); });

  /* ---------------------------------------------------------------- confirm modal */
  let confirmId = null;
  function showConfirm(m) { confirmId = m.id; $('confirm-name').textContent = m.name; $('confirm-summary').textContent = m.summary; $('confirm').classList.remove('hidden'); }
  function hideConfirm() { confirmId = null; $('confirm').classList.add('hidden'); }
  $('confirm-yes').onclick = () => { send({ type: 'confirm_reply', id: confirmId, approved: true }); hideConfirm(); };
  $('confirm-no').onclick = () => { send({ type: 'confirm_reply', id: confirmId, approved: false }); hideConfirm(); };

  connect();
})();
