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

  /* ---------------------------------------------------------------- tabs, prompt & full log */
  document.querySelectorAll('nav.tabs .tab').forEach((b) => b.onclick = () => {
    document.querySelectorAll('nav.tabs .tab').forEach((x) => x.classList.toggle('active', x === b));
    for (const id of ['chat', 'prompt', 'log']) $(id).classList.toggle('hidden', id !== b.dataset.tab);
    document.querySelector('footer').classList.toggle('hidden', b.dataset.tab !== 'chat');
    $('attachments').classList.toggle('hidden', b.dataset.tab !== 'chat');
  });
  let logCount = 0;
  const logList = $('log-list');
  function logEntry(cls, title, bodyHtml, open) {
    const d = document.createElement('details'); d.className = 'logent ' + cls; if (open) d.open = true;
    d.innerHTML = `<summary>${title}</summary>${bodyHtml}`; logList.appendChild(d);
    logCount++; $('log-count').textContent = `(${logCount})`;
    if (logList.children.length > 400) logList.removeChild(logList.firstChild);
    return d;
  }
  function renderMessages(msgs) {
    return msgs.map((m) => {
      let body = '';
      if (typeof m.content === 'string' && m.content) body += esc(m.content);
      else if (m.content && typeof m.content !== 'string') body += esc(JSON.stringify(m.content, null, 1));
      if (m.tool_calls) body += (body ? '\n' : '') + '⚙ tool_calls: ' + esc(JSON.stringify(m.tool_calls.map((t) => ({ id: t.id, name: t.function?.name, arguments: t.function?.arguments })), null, 1));
      const extra = m.tool_call_id ? ` <span class="muted">(tool_call_id ${esc(m.tool_call_id)}${m.name ? ', ' + esc(m.name) : ''})</span>` : '';
      return `<div class="m ${esc(m.role)}"><span class="msg-role">${esc(m.role)}</span>${extra}<pre>${body}</pre></div>`;
    }).join('');
  }
  function handleTrace(m) {
    const t = new Date().toLocaleTimeString();
    if (m.kind === 'request') {
      const sys = (m.messages || []).find((x) => x.role === 'system');
      if (sys) { $('prompt-text').textContent = sys.content; $('prompt-meta').textContent = `${t} · ход ${m.turn}, раунд ${m.round} · ${m.model} · инструменты: ${(m.tools || []).join(', ') || 'нет'} · ${JSON.stringify(m.params)}`; }
      const who = m.agent === 'executor' ? '🛠 исполнитель' : '🗣 голосовой агент';
      logEntry('req', `→ <b>запрос</b> ${t} · ${who}${m.stage ? ' / ' + esc(m.stage) : ''} · ход ${m.turn} · вызов ${m.round} · ${esc(m.model)} · сообщений: ${m.messages.length} · инструменты: ${esc((m.tools || []).join(', ') || 'нет')} · ${esc(JSON.stringify(m.params))}`, renderMessages(m.messages), false);
    } else if (m.kind === 'response') {
      const who = m.agent === 'executor' ? '🛠 исполнитель' : '🗣 голосовой агент';
      const tc = (m.tool_calls || []).map((c) => `${c.function?.name}(${c.function?.arguments})`).join('\n');
      const body = `<pre>${m.reasoning ? '🧠 reasoning:\n' + esc(m.reasoning) + '\n\n' : ''}${esc(m.content || '')}${tc ? '\n⚙ tool_calls:\n' + esc(tc) : ''}\n\nfinish_reason: ${esc(String(m.finish_reason))} · usage: ${esc(JSON.stringify(m.usage))} · ${m.ms} мс</pre>`;
      logEntry('res', `← <b>ответ</b> ${t} · ${who} · ход ${m.turn} · вызов ${m.round} · ${(m.content || '').length} симв. · ${(m.tool_calls || []).length} вызов. · ${m.ms} мс`, body, false);
    }
  }
  let execCard = null;
  $('log-clear').onclick = () => { logList.innerHTML = ''; logCount = 0; $('log-count').textContent = ''; };

  /* ---------------------------------------------------------------- prompt editor */
  const PROMPT_KEYS = ['system', 'voice_prose', 'voice_text', 'executor'];
  let promptState = null;
  function renderPrompts(p) {
    promptState = p;
    for (const k of PROMPT_KEYS) {
      const ta = $('ed-' + k); ta.value = p.current[k]; ta.classList.remove('dirty');
      $('ov-' + k).textContent = (p.overridden || []).includes(k) ? '· изменён' : '· по умолчанию';
    }
    $('prompt-status').textContent = p.saved ? 'сохранено ' + new Date().toLocaleTimeString() : '';
  }
  for (const k of PROMPT_KEYS) $('ed-' + k).addEventListener('input', (e) => { e.target.classList.toggle('dirty', promptState && e.target.value !== promptState.current[k]); });
  $('prompt-save').onclick = () => { const values = {}; for (const k of PROMPT_KEYS) values[k] = $('ed-' + k).value; send({ type: 'set_prompts', values }); $('prompt-status').textContent = 'сохраняю…'; };
  $('prompt-reset').onclick = () => { if (confirm('Вернуть все три части к встроенным значениям?')) send({ type: 'reset_prompts' }); };
  $('prompt-reload').onclick = () => send({ type: 'get_prompts' });

  function handle(m) {
    switch (m.type) {
      case 'status': {
        const wasConnected = state && state.server; state = m; renderStatus();
        if (m.server && (!wasConnected || !promptState)) send({ type: 'get_prompts' });
        break;
      }
      case 'prompts': renderPrompts(m); break;
      case 'trace': handleTrace(m); break;
      case 'stage': current = null; reasoningCard = null; execCard = null; break;
      case 'task': card('task', `🎯 <b>задача исполнителю</b>`, m.task, true); current = null; break;
      case 'executor_delta': {
        if (!execCard) execCard = card('tool exec', '🛠 <b>исполнитель</b>', '', false);
        execCard.querySelector('pre').textContent += m.content; break;
      }
      case 'report': card('report', `📋 <b>отчёт исполнителя</b> · ${(m.report || '').length} симв.`, m.report, true); current = null; break;
      case 'user_message': {
        const d = div('msg user');
        const src = m.source === 'voice' ? '🎙 голос' : '⌨ текст';
        let html = `<div class="src">${src}${m.memory ? ' · 🗂 память' : ''}</div>${fmt(m.text || '')}`;
        if (m.attachments && m.attachments.length) html += `<div class="src">📎 ${m.attachments.length} влож.</div>`;
        d.innerHTML = html; add(d);
        current = null; reasoningCard = null; metrics = { stt: metrics.stt }; updateMetrics();
        break;
      }
      case 'round': if (m.round > 1) { current = null; } break;
      case 'delta': {
        if (!current) { current = add(div('msg assistant streaming')); current._text = ''; }
        current._text += m.content;
        current.innerHTML = (current._speech ? '<span class="spk" title="озвучено">🔊</span>' : '') + fmt(current._text); scroll();
        break;
      }
      case 'speech_delta': {
        // TTS-ready text from the `speak` tool: show it while it streams; `display` may replace it at the end
        if (!current || !current._speech) { current = add(div('msg assistant streaming speech')); current._text = ''; current._speech = true; }
        current._text += m.content; current.innerHTML = '<span class="spk" title="озвучено">🔊</span>' + fmt(current._text); scroll();
        break;
      }
      case 'speech_done': {
        if (current && current._speech) {
          if (m.display) { current._text = m.display; current.innerHTML = '<span class="spk" title="озвучено (на экране — версия для чтения)">🔊</span>' + fmt(m.display); }
          current.classList.remove('streaming');
          if (!m.final) current = null;
        }
        break;
      }
      case 'reasoning': {
        if (!reasoningCard) { reasoningCard = card('reasoning', '🧠 рассуждения', '', false); }
        const pre = reasoningCard.querySelector('pre'); pre.textContent += m.content; break;
      }
      case 'tool_call':
        card('tool', `🔧 <b>${esc(m.name)}</b>`, JSON.stringify(m.arguments ?? m.raw, null, 1), false); current = null;
        logEntry('tool', `⚙ <b>вызов ${esc(m.name)}</b> ${new Date().toLocaleTimeString()}`, `<pre>${esc(JSON.stringify(m.arguments ?? m.raw, null, 1))}</pre>`, false);
        break;
      case 'tool_result': {
        const r = m.result || {}; const ok = !r.error;
        card('tool', `${ok ? '✅' : '⚠️'} <b>${esc(m.name)}</b> · ${m.ms} мс`, JSON.stringify(r, null, 1).slice(0, 4000), !ok);
        logEntry('tool', `${ok ? '✅' : '⚠️'} <b>результат ${esc(m.name)}</b> · ${m.ms} мс`, `<pre>${esc(JSON.stringify(r, null, 1))}</pre>`, false);
        break;
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
    setPill('pill-vision', si.vision ? 'ok' : 'warn', si.vision ? 'omni' : 'omni off');
    const mem = si.memory ? Object.values(si.memory).reduce((a, b) => a + b, 0) : 0;
    setPill('pill-memory', 'ok', `память ${mem}`);
    $('model').textContent = si.model ? '· ' + si.model.split('/').pop() : '';
    $('btn-listen').classList.toggle('active', !!state.listening);
    $('btn-memory').classList.toggle('active', !!(state.settings && state.settings.memory_recall));
    $('pill-stt').title = state.mic ? 'микрофон: ' + state.mic : '';
    const s = state.settings || {};
    for (const k of ['tts_mode', 'confirm', 'stt_language']) $('s-' + k).value = s[k];
    for (const k of ['auto_listen', 'barge_in', 'tools_enabled']) $('s-' + k).checked = !!s[k];
    fillVoices('s-voice_ru', state.voices.ru, s.voice_ru); fillVoices('s-voice_en', state.voices.en, s.voice_en);
    fillDevices('s-speaker_device', state.output_devices || [], s.speaker_device);
    $('speaker-now').textContent = state.speaker ? 'сейчас: ' + state.speaker : '';
    fillDevices('s-mic_device', state.input_devices || [], s.mic_device);
    $('mic-now').textContent = state.microphone ? 'сейчас: ' + state.microphone + (state.mic && state.mic !== 'ready' ? ' · ' + state.mic : '') : (state.mic || '');
    $('s-tts_speed').value = s.tts_speed; $('s-tts_speed-v').textContent = Number(s.tts_speed).toFixed(2);
    $('settings-status').textContent = `сессия ${state.session_id || '—'} · mic ${state.mic || ''}`;
  }
  function fillDevices(id, devs, val) {
    const sel = $(id);
    const sig = devs.map((d) => d.index).join(',');
    if (sel._sig !== sig) { sel.innerHTML = devs.map((d) => `<option value="${d.index === null ? '' : d.index}">${esc(d.name)}${d.api ? ' · ' + esc(d.api.replace('Windows ', '')) : ''}</option>`).join(''); sel._sig = sig; }
    sel.value = val || '';
  }
  function fillVoices(id, list, val) {
    const sel = $(id); if (sel.options.length !== list.length) { sel.innerHTML = list.map((v) => `<option value="${v}">${v}</option>`).join(''); }
    sel.value = val;
  }
  function pushSettings(patch) { send({ type: 'settings', settings: patch }); }
  $('btn-settings').onclick = () => $('settings').classList.toggle('hidden');
  for (const k of ['tts_mode', 'confirm', 'stt_language', 'voice_ru', 'voice_en', 'speaker_device', 'mic_device']) $('s-' + k).onchange = (e) => pushSettings({ [k]: e.target.value });
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
  $('btn-memory').onclick = () => pushSettings({ memory_recall: !(state?.settings?.memory_recall) });

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
