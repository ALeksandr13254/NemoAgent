/* NemoAgent client UI — vanilla JS, talks to the local client process over /ui WebSocket. */
(() => {
  const $ = (id) => document.getElementById(id);
  const chat = $('chat'), input = $('input'), attBox = $('attachments');
  let ws = null, state = null, pending = [];        // pending attachments [{id,name,is_image,...}]
  let current = null;                                // current assistant bubble
  const endCurrent = () => { if (current) current.classList.remove('streaming'); current = null; };  // the bubble stops receiving text
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
  function send(msg) {
    if (ws && ws.readyState === 1) { ws.send(JSON.stringify(msg)); return; }
    // no link to the local client process: say so instead of dropping the message silently
    if (msg.type === 'send' || msg.type === 'say') add(div('errline', '⚠ нет связи с клиентом NemoAgent (окно run_client.bat) — страница переподключается, обновите её (Ctrl+F5), если это не проходит'));
  }
  function setPill(id, cls, text) { const p = $(id); p.className = 'pill ' + cls; if (text) p.textContent = text; }
  /* ---------------------------------------------------------------- icons (inline SVG: crisp and readable, unlike the OS emoji font) */
  const SVG = (inner, fill) => `<svg viewBox="0 0 24 24" fill="${fill || 'none'}" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${inner}</svg>`;
  const ICONS = {
    paperclip: SVG('<path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>'),
    screenshot: SVG('<rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/><circle cx="12" cy="10" r="3"/>'),
    memory: SVG('<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>'),
    mic: SVG('<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><path d="M12 19v3M8 22h8"/>'),
    listen: SVG('<path d="M4 10v4M8 6v12M12 3v18M16 7v10M20 10v4"/>'),
    stop: SVG('<rect x="5" y="5" width="14" height="14" rx="2"/>', 'currentColor'),
    send: SVG('<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>'),
    speaker: SVG('<path d="M11 5 6 9H2v6h4l5 4V5Z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M19 5a10 10 0 0 1 0 14"/>'),
    gear: SVG('<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>'),
    menu: SVG('<path d="M4 6h16M4 12h16M4 18h16"/>'),
    record: SVG('<circle cx="12" cy="12" r="7"/>', 'currentColor'),
    play: SVG('<path d="M6 4v16l14-8Z"/>', 'currentColor'),
  };
  const ICON_BUTTONS = {
    'btn-attach': ['paperclip', 'Прикрепить файлы: картинки, аудио, видео, документы'],
    'btn-shot': ['screenshot', 'Снять скриншот экрана и прикрепить к сообщению'],
    'btn-memory': ['memory', 'Память (RAG): пока включено, к сообщениям подмешиваются воспоминания из прошлых разговоров'],
    'btn-ptt': ['mic', 'Говорить: удерживайте кнопку или пробел (push-to-talk)'],
    'btn-listen': ['listen', 'Слушать автоматически: голосовая активность (VAD) сама находит фразу'],
    'btn-stop': ['stop', 'Остановить ответ и озвучку'],
    'btn-send': ['send', 'Отправить (Enter)'],
    'btn-settings': ['gear', 'Настройки'],
    'btn-history': ['menu', 'Показать или спрятать левую панель'],
  };
  for (const [id, [name, title]] of Object.entries(ICON_BUTTONS)) { const b = $(id); if (b) { b.innerHTML = ICONS[name]; b.title = title; } }
  $('reader-play').innerHTML = ICONS.play + ' Озвучить'; $('reader-stop').innerHTML = ICONS.stop + ' Стоп';
  $('reader-dictate').innerHTML = ICONS.mic + ' Диктовка';

  /* speaker button on every bubble: click = read this message aloud (again) with the local TTS */
  function spkBtn(spoken, title) {
    return `<button class="spk-btn${spoken ? ' spoken' : ''}" title="${title || (spoken ? 'озвучено · нажмите, чтобы повторить' : 'озвучить')}">${ICONS.speaker}</button>`;
  }
  chat.addEventListener('click', (e) => {
    const b = e.target.closest('.spk-btn'); if (!b) return;
    const msg = b.closest('.msg'); if (!msg) return;
    const text = (msg._spoken || msg._text || msg.innerText || '').trim();
    if (text) send({ type: 'say', text });
  });
  /* waiting indicator: seconds since the message went out, until the model's first token */
  let waitTimer = null;
  function waitingSince(t0) {
    clearInterval(waitTimer);
    if (!t0) { waitTimer = null; return; }
    const tick = () => { const s = Math.round((Date.now() - t0) / 1000); if (s >= 2) $('stt-state').textContent = `модель думает… ${s} с`; };
    waitTimer = setInterval(tick, 1000);
  }
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
    setPill('pill-server', 'warn', 'подключаюсь…');
    try {
      ws = new WebSocket(`ws://${location.host}/ui`);
    } catch (err) {
      console.error('WebSocket to the client failed', err);
      setPill('pill-server', 'err', 'нет клиента'); setTimeout(connect, 3000); return;
    }
    ws.onopen = () => { send({ type: 'get_state' }); };
    ws.onerror = (e) => console.error('client websocket error', e);
    ws.onclose = () => { setPill('pill-server', 'err', 'нет клиента'); setTimeout(connect, 1500); };
    ws.onmessage = (e) => { try { handle(JSON.parse(e.data)); } catch (err) { console.error(err, e.data); } };
  }
  // connect first: a later script error must not leave the page without its link to the client
  connect();

  /* ---------------------------------------------------------------- tabs, prompt & full log */
  function activeTab() { return document.querySelector('nav.tabs .tab.active')?.dataset.tab || 'chat'; }
  document.querySelectorAll('nav.tabs .tab').forEach((b) => b.onclick = () => {
    document.querySelectorAll('nav.tabs .tab').forEach((x) => x.classList.toggle('active', x === b));
    for (const id of ['chat', 'prompt', 'log', 'reader', 'memory']) $(id).classList.toggle('hidden', id !== b.dataset.tab);
    if (b.dataset.tab === 'memory') memRefresh();
    document.querySelector('footer').classList.toggle('hidden', b.dataset.tab !== 'chat');
    $('attachments').classList.toggle('hidden', b.dataset.tab !== 'chat');
    updateSidebar();
  });
  let logCount = 0;
  const logList = $('log-list');
  const agentLabel = (a) => a === 'executor' ? '🛠 исполнитель' : a === 'router' ? '🧭 маршрутизатор' : '🗣 голосовой агент';
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
      // the "last prompt" block follows the two agents; the router's parallel call is only in the log
      if (sys && m.agent !== 'router') { $('prompt-text').textContent = sys.content; $('prompt-meta').textContent = `${t} · ход ${m.turn}, раунд ${m.round} · ${m.model} · инструменты: ${(m.tools || []).join(', ') || 'нет'} · ${JSON.stringify(m.params)}`; }
      const who = agentLabel(m.agent);
      logEntry('req', `→ <b>запрос</b> ${t} · ${who}${m.stage ? ' / ' + esc(m.stage) : ''} · ход ${m.turn} · вызов ${m.round} · ${esc(m.model)} · сообщений: ${m.messages.length} · инструменты: ${esc((m.tools || []).join(', ') || 'нет')} · ${esc(JSON.stringify(m.params))}`, renderMessages(m.messages), false);
    } else if (m.kind === 'response') {
      const who = agentLabel(m.agent);
      const tc = (m.tool_calls || []).map((c) => `${c.function?.name}(${c.function?.arguments})`).join('\n');
      const body = `<pre>${m.reasoning ? '🧠 reasoning:\n' + esc(m.reasoning) + '\n\n' : ''}${esc(m.content || '')}${tc ? '\n⚙ tool_calls:\n' + esc(tc) : ''}\n\nfinish_reason: ${esc(String(m.finish_reason))} · usage: ${esc(JSON.stringify(m.usage))} · ${m.ms} мс</pre>`;
      logEntry('res', `← <b>ответ</b> ${t} · ${who} · ход ${m.turn} · вызов ${m.round} · ${(m.content || '').length} симв. · ${(m.tool_calls || []).length} вызов. · ${m.ms} мс`, body, false);
    }
  }
  let execCard = null;
  $('log-clear').onclick = () => { logList.innerHTML = ''; logCount = 0; $('log-count').textContent = ''; };

  /* ---------------------------------------------------------------- prompt editor */
  const PROMPT_KEYS = ['system', 'voice_prose', 'voice_text', 'executor', 'router'];
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
        if (!wasConnected) send({ type: 'chats' });
        break;
      }
      case 'chats': renderChats(m); break;
      case 'chat_loaded': renderHistory(m.chat); break;
      case 'memory_stats': {
        const total = m.memory ? Object.values(m.memory).reduce((a, b) => a + b, 0) : 0;
        const what = m.what === 'clear' ? 'память очищена' : m.what === 'prune' ? 'прибрано' : 'записи чата забыты';
        $('mem-status').textContent = `${what}: удалено ${m.removed}, осталось ${total}`;
        if (!$('memory').classList.contains('hidden')) memRefresh();
        break;
      }
      case 'memory_items': memRender(m); break;
      case 'memory_saved': {
        const label = m.action === 'add' ? (m.ok ? 'добавлено' : 'не добавлено') : m.action === 'update' ? (m.ok ? 'сохранено' : 'не сохранено') : (m.ok ? 'удалено' : 'не удалено');
        $('mem-ed-status').textContent = label + ' ' + new Date().toLocaleTimeString();
        $('mem-status').textContent = label + ' ' + new Date().toLocaleTimeString();
        if (m.action === 'add' && m.ok) { memSel = m.id; memText._id = null; memQuery = ''; $('mem-search').value = ''; }
        if (m.action === 'update' && m.ok) { memText.classList.remove('dirty'); $('mem-ed-save').disabled = true; }
        if (m.action === 'delete' && m.ok) memClose();
        memRefresh();
        break;
      }
      case 'prompts': renderPrompts(m); break;
      case 'trace': handleTrace(m); break;
      case 'stage': endCurrent(); reasoningCard = null; execCard = null; break;
      case 'task': card('task', `🎯 <b>задача исполнителю</b>`, m.task, true); endCurrent(); break;
      case 'executor_delta': {
        if (!execCard) execCard = card('tool exec', '🛠 <b>исполнитель</b>', '', false);
        execCard.querySelector('pre').textContent += m.content; break;
      }
      case 'report': card('report', `📋 <b>отчёт исполнителя</b> · ${(m.report || '').length} симв.`, m.report, true); endCurrent(); break;
      case 'user_message': {
        const d = div('msg user'); d._text = m.text || '';
        const src = m.source === 'voice' ? '🎙 голос' : '⌨ текст';
        let html = `<div class="src">${src}${m.memory ? ' · 🗂 память' : ''}${spkBtn(false, 'озвучить это сообщение')}</div>${fmt(m.text || '')}`;
        if (m.attachments && m.attachments.length) html += `<div class="src">📎 ${m.attachments.length} влож.</div>`;
        d.innerHTML = html; add(d);
        current = null; reasoningCard = null; metrics = { stt: metrics.stt }; updateMetrics();
        waitingSince(Date.now());   // "думаю… N с" until the first token: the free pool can take 20 s
        break;
      }
      case 'round': if (m.round > 1) endCurrent(); break;
      case 'delta': {
        if (waitTimer) { waitingSince(null); $('stt-state').textContent = ''; }
        if (!current) { current = add(div('msg assistant streaming')); current._text = ''; }
        current._text += m.content;
        current.innerHTML = spkBtn(current._speech) + fmt(current._text); scroll();
        break;
      }
      case 'speech_delta': {
        if (waitTimer) { waitingSince(null); $('stt-state').textContent = ''; }
        // TTS-ready text: show it while it streams; `display` may replace it on screen at the end,
        // the spoken version is kept for the 🔊 button
        if (!current || !current._speech) { current = add(div('msg assistant streaming speech')); current._text = ''; current._spoken = ''; current._speech = true; }
        current._text += m.content; current._spoken += m.content;
        current.innerHTML = spkBtn(true) + fmt(current._text); scroll();
        break;
      }
      case 'speech_done': {
        if (current && current._speech) {
          if (m.display) {
            // the screen-only part (code, paths, exact figures after ===) is shown UNDER the spoken text, not instead of it
            current._display = m.display; current._text = current._spoken + '\n' + m.display;
            current.innerHTML = spkBtn(true) + fmt(current._spoken) + `<div class="display">${fmt(m.display)}</div>`;
          }
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
        card('tool', `🔧 <b>${esc(m.name)}</b>`, JSON.stringify(m.arguments ?? m.raw, null, 1), false); endCurrent();
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
      case 'error': {
        const text = (m.message || '').trim() || (m.detail || '').trim() || 'ошибка без описания (см. окно сервера)';
        const e = add(div('errline', '⚠ ' + esc(text))); if (m.detail) e.title = m.detail;
        logEntry('error', `⚠ <b>ошибка</b> ${new Date().toLocaleTimeString()}`, `<pre>${esc(text)}${m.detail && m.detail !== text ? '\n' + esc(m.detail) : ''}</pre>`, true);
        break;
      }
      case 'done': {
        waitingSince(null);
        if (current) current.classList.remove('streaming');
        if (m.first_token_ms) metrics.first_token = m.first_token_ms;
        if (m.total_ms) metrics.total = m.total_ms; updateMetrics();
        if (m.finish_reason === 'interrupted') add(div('notice', 'прервано'));
        $('stt-state').textContent = ''; current = null; break;
      }
      case 'tts_first_audio': metrics.first_audio = m.ms; updateMetrics(); break;
      case 'interrupted': if (current) current.classList.remove('streaming'); break;
      case 'cleared': chat.innerHTML = ''; current = null; reasoningCard = null; break;
      case 'mic': {
        // the same level bar lives under the chat composer and in the read-aloud tab (dictation)
        for (const el of [$('mic-level'), $('reader-mic-level')]) { el.style.width = Math.round(m.level * 100) + '%'; el.classList.toggle('speech', !!m.speech); }
        break;
      }
      case 'stt': {
        let text = '';
        if (m.state === 'transcribing') text = `распознаю ${m.duration} с…`;
        else if (m.state === 'done') { text = ''; metrics.stt = m.ms; }
        else if (m.state === 'empty') text = 'ничего не распознано';
        else if (m.state === 'error') text = 'ошибка STT: ' + m.message;
        $('stt-state').textContent = text; $('reader-stt').textContent = text;
        break;
      }
      case 'confirm': showConfirm(m); break;
      case 'confirm_expired': hideConfirm(); break;
      case 'dictation': readerAppend(m.text); break;
      case 'read': readerProgress(m); break;
    }
  }

  /* ---------------------------------------------------------------- read-aloud tab */
  const READER_KEY = 'nemoagent-reader-text';
  const rt = $('reader-text');
  try { rt.value = localStorage.getItem(READER_KEY) || ''; } catch (e) { /* storage unavailable */ }
  let readerTotal = 0, readerSaveTimer = null;
  function readerCount() {
    const n = rt.value.length, words = (rt.value.match(/\S+/g) || []).length;
    const sec = Math.round(n / 14);   // TeraTTS at speed 1.0 reads roughly fourteen characters a second
    $('reader-count').textContent = n ? `${n} симв. · ${words} слов · ≈ ${sec >= 60 ? Math.floor(sec / 60) + ' мин ' + (sec % 60) + ' с' : sec + ' с'}` : '';
  }
  function readerSave() {
    clearTimeout(readerSaveTimer);
    readerSaveTimer = setTimeout(() => { try { localStorage.setItem(READER_KEY, rt.value); } catch (e) { /* ignore */ } }, 300);
  }
  function readerAppend(text) {
    if (!text) return;
    const v = rt.value, sep = !v ? '' : /\s$/.test(v) ? '' : ' ';
    rt.value = v + sep + text; rt.scrollTop = rt.scrollHeight; readerCount(); readerSave();
  }
  function readerProgress(m) {
    const bar = $('reader-bar'), now = $('reader-now');
    if (m.state === 'start') { readerTotal = m.total || 0; bar.style.width = '0%'; now.textContent = `синтезирую… (${readerTotal} предл.)`; $('reader-play').classList.add('active'); }
    else if (m.state === 'sentence') {
      const total = m.total || readerTotal || 1;
      bar.style.width = Math.min(100, Math.round(100 * m.index / total)) + '%';
      now.textContent = `${m.index} / ${total}: ${m.text}`;
    }
    else if (m.state === 'done') { bar.style.width = '100%'; now.textContent = 'готово'; $('reader-play').classList.remove('active'); }
    else if (m.state === 'stopped') { bar.style.width = '0%'; now.textContent = 'остановлено'; $('reader-play').classList.remove('active'); }
    else if (m.state === 'error') { now.textContent = '⚠ ' + (m.message || 'озвучка недоступна'); $('reader-play').classList.remove('active'); }
    if (typeof m.dictation === 'boolean') readerDictation(m.dictation);
  }
  function readerDictation(on) {
    $('reader-dictate').classList.toggle('active', !!on);
    rt.classList.toggle('dictating', !!on);
    if (on) $('reader-now').textContent = 'диктовка: говорите, распознанные фразы добавляются в текст';
  }
  rt.addEventListener('input', () => { readerCount(); readerSave(); });
  $('reader-play').onclick = () => {
    const a = rt.selectionStart, b = rt.selectionEnd;
    const text = (a !== b ? rt.value.slice(a, b) : rt.value).trim();
    if (!text) { $('reader-now').textContent = 'текст пустой'; return; }
    send({ type: 'read', text });
  };
  $('reader-stop').onclick = () => send({ type: 'read_stop' });
  $('reader-clear').onclick = () => { if (!rt.value || confirm('Очистить текст?')) { rt.value = ''; readerCount(); readerSave(); } };
  $('reader-dictate').onclick = () => send({ type: 'dictate', on: !$('reader-dictate').classList.contains('active') });
  readerCount();

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
    if (!memQuery) $('mem-total').textContent = String(mem);
    $('model').textContent = si.model ? '· ' + si.model.split('/').pop() : '';
    $('btn-listen').classList.toggle('active', !!state.listening);
    readerDictation(!!state.dictation);
    $('btn-memory').classList.toggle('active', !!(state.settings && state.settings.memory_recall));
    $('pill-stt').title = state.mic ? 'микрофон: ' + state.mic : '';
    const s = state.settings || {};
    for (const k of ['tts_mode', 'confirm', 'stt_language', 'tts_language']) $('s-' + k).value = s[k] || (k === 'tts_language' ? 'ru' : s[k]);
    const models = si.models || {};
    for (const role of ['dialogue', 'executor', 'router', 'media']) $('s-model_' + role).value = s['model_' + role] || models[role] || '';
    $('model').title = Object.entries(models).map(([r, m]) => `${r}: ${m}`).join('\n');
    for (const k of ['auto_listen', 'barge_in', 'tools_enabled']) $('s-' + k).checked = !!s[k];
    fillVoices('s-voice_ru', state.voices.ru, s.voice_ru); fillVoices('s-voice_en', state.voices.en, s.voice_en);
    fillDevices('s-speaker_device', state.output_devices || [], s.speaker_device);
    $('speaker-now').textContent = state.speaker ? 'сейчас: ' + state.speaker : '';
    fillDevices('s-mic_device', state.input_devices || [], s.mic_device);
    $('mic-now').textContent = state.microphone ? 'сейчас: ' + state.microphone + (state.mic && state.mic !== 'ready' ? ' · ' + state.mic : '') : (state.mic || '');
    $('s-tts_speed').value = s.tts_speed; $('s-tts_speed-v').textContent = Number(s.tts_speed).toFixed(2);
    renderMonitors(state.monitors || [], s.screenshot_monitors || []);
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
  /* which monitors the screenshot button / look_at_screen capture: "all" (an empty list) or a set of monitor numbers */
  function renderMonitors(mons, chosen) {
    const box = $('s-monitors');
    const all = !chosen.length;
    const sig = mons.map((m) => `${m.index}:${m.width}x${m.height}`).join(',') + '|' + chosen.join(',');
    if (box._sig === sig) return;
    box._sig = sig;
    if (!mons.length) { box.innerHTML = '<div class="muted small">мониторы не найдены</div>'; return; }
    const row = (val, label, checked, disabled) => `<label><input type="checkbox" data-mon="${val}"${checked ? ' checked' : ''}${disabled ? ' disabled' : ''}> ${label}</label>`;
    box.innerHTML = row('all', 'все мониторы', all, false)
      + mons.map((m) => row(m.index, `монитор ${m.index}${m.primary ? ' (основной)' : ''} · ${m.width}×${m.height}`, all || chosen.includes(m.index), all)).join('');
    box.querySelectorAll('input').forEach((inp) => {
      inp.onchange = () => {
        let list;
        if (inp.dataset.mon === 'all') list = inp.checked ? [] : mons.slice(0, 1).map((m) => m.index);   // "all" off: start from the first monitor
        else {
          list = [...box.querySelectorAll('input[data-mon]')].filter((b) => b.dataset.mon !== 'all' && b.checked).map((b) => Number(b.dataset.mon));
          if (!list.length || list.length === mons.length) list = [];   // nothing or everything ticked = all monitors
        }
        box._sig = null; pushSettings({ screenshot_monitors: list });
      };
    });
  }
  // the settings column on the right is pinned like the chats sidebar; ⚙ hides or shows it, the choice is remembered
  const SETTINGS_KEY = 'nemoagent-settings-open';
  let settingsHidden = false;
  try { settingsHidden = localStorage.getItem(SETTINGS_KEY) === '0'; } catch (e) { /* ignore */ }
  function updateSettingsPane() { $('settings').classList.toggle('collapsed', settingsHidden); $('btn-settings').classList.toggle('active', !settingsHidden); }
  $('btn-settings').onclick = () => {
    settingsHidden = !settingsHidden;
    try { localStorage.setItem(SETTINGS_KEY, settingsHidden ? '0' : '1'); } catch (e) { /* ignore */ }
    updateSettingsPane();
  };
  updateSettingsPane();
  for (const k of ['tts_mode', 'confirm', 'stt_language', 'tts_language', 'model_dialogue', 'model_executor', 'model_router', 'model_media', 'voice_ru', 'voice_en', 'speaker_device', 'mic_device']) $('s-' + k).onchange = (e) => pushSettings({ [k]: e.target.value });
  for (const k of ['auto_listen', 'barge_in', 'tools_enabled']) $('s-' + k).onchange = (e) => pushSettings({ [k]: e.target.checked });
  $('s-tts_speed').oninput = (e) => { $('s-tts_speed-v').textContent = Number(e.target.value).toFixed(2); };
  $('s-tts_speed').onchange = (e) => pushSettings({ tts_speed: Number(e.target.value) });
  $('btn-say').onclick = () => {
    // the test phrase follows the voice language: the Russian voice reads Russian, the English one English,
    // "auto" reads both so you can hear the switch between them
    const lang = $('s-tts_language').value || 'auto';
    const text = lang === 'ru' ? 'Привет! Голос работает, всё в порядке.'
      : lang === 'en' ? 'Hello! The voice is working, everything is fine.'
        : 'Привет! Голос работает. Hello, the voice is working.';
    send({ type: 'say', text });
  };

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
      if (m.error) { add(div('errline', '⚠ ' + esc(m.error))); pending = pending.filter((p) => p !== entry); }
      else {   // one attachment per captured monitor
        const list = (m.attachments || [m]).map((a) => Object.assign({}, a, { pending: false, is_image: true }));
        pending.splice(pending.indexOf(entry), 1, ...list);
        if (m.warning) add(div('errline', '⚠ ' + esc(m.warning)));
      }
      renderAttachments(); } };
    ws.addEventListener('message', h); send({ type: 'screenshot', request_id: rid });
  };
  $('btn-stop').onclick = () => send({ type: 'interrupt' });
  $('btn-new').onclick = () => send({ type: 'new_session' });

  /* ---------------------------------------------------------------- memory tab: list on the left, one record in the middle */
  const memItemsEl = $('mem-items'), memText = $('mem-ed-text');
  let memQuery = '', memItems = [], memSel = null;   // memSel: record id, 'new', or null
  function memRefresh() { if (memQuery) send({ type: 'memory_search', query: memQuery }); else send({ type: 'memory_list' }); }
  const KIND_RU = { dialog: 'диалог', media: 'с картинками', note: 'заметка', analysis: 'анализ' };
  const memWhen = (ts) => new Date(ts * 1000).toLocaleString([], { day: '2-digit', month: '2-digit', year: '2-digit', hour: '2-digit', minute: '2-digit' });
  function memTitle(text) {
    const t = (text || '').replace(/^Attachments:[^\n]*\n/, '').replace(/^User:\s*/, '').split('\n')[0].trim();
    return t.length > 70 ? t.slice(0, 70) + '…' : t || '(пусто)';
  }
  function memRender(m) {
    memItems = m.items || [];
    $('mem-total').textContent = m.query ? `найдено ${memItems.length} из ${m.total}` : `${m.total}`;
    memItemsEl.innerHTML = memItems.length ? '' : `<div id="hist-empty">${m.query ? 'ничего похожего' : 'память пуста'}</div>`;
    for (const it of memItems) {
      const el = div('hist-item' + (it.id === memSel ? ' active' : '')); el.dataset.id = it.id;
      const score = it.score != null ? ` · ${Number(it.score).toFixed(2)}` : '';
      el.innerHTML = `<div class="t">${esc(memTitle(it.text))}</div><div class="d">${memWhen(it.ts)} · ${KIND_RU[it.kind] || esc(it.kind || '')}${it.collection === 'vl' ? ' · 🖼' : ''}${score}</div>`;
      memItemsEl.appendChild(el);
    }
    if (typeof memSel === 'number') {
      const it = memItems.find((x) => x.id === memSel);
      if (it) memOpen(it, true); else memClose();
    }
  }
  function memShowEditor(show) { $('mem-editor').classList.toggle('hidden', !show); $('mem-empty').classList.toggle('hidden', show); }
  function memOpen(it, keepText) {
    memSel = it.id;
    memItemsEl.querySelectorAll('.hist-item').forEach((x) => x.classList.toggle('active', Number(x.dataset.id) === it.id));
    $('mem-ed-title').textContent = `Воспоминание #${it.id}`;
    $('mem-ed-meta').textContent = `${memWhen(it.ts)} · ${KIND_RU[it.kind] || it.kind || ''} · ${it.collection === 'vl' ? 'коллекция с картинками' : 'текстовая коллекция'}${it.score != null ? ' · близость ' + Number(it.score).toFixed(2) : ''}`;
    if (!keepText || memText._id !== it.id) { memText.value = it.text || ''; memText._orig = memText.value; memText._id = it.id; memText.classList.remove('dirty'); $('mem-ed-save').disabled = true; $('mem-ed-status').textContent = ''; }
    $('mem-ed-del').classList.remove('hidden');
    memShowEditor(true);
  }
  function memNew() {
    memSel = 'new';
    memItemsEl.querySelectorAll('.hist-item').forEach((x) => x.classList.remove('active'));
    $('mem-ed-title').textContent = 'Новое воспоминание'; $('mem-ed-meta').textContent = 'факт о вас, предпочтение, договорённость — станет записью вида «заметка»';
    memText.value = ''; memText._orig = null; memText._id = null; memText.classList.remove('dirty'); $('mem-ed-save').disabled = true; $('mem-ed-status').textContent = '';
    $('mem-ed-del').classList.add('hidden');
    memShowEditor(true); memText.focus();
  }
  function memClose() { memSel = null; memShowEditor(false); memItemsEl.querySelectorAll('.hist-item').forEach((x) => x.classList.remove('active')); }
  memText.addEventListener('input', () => {
    const dirty = memSel === 'new' ? !!memText.value.trim() : memText.value !== memText._orig;
    memText.classList.toggle('dirty', dirty && memSel !== 'new'); $('mem-ed-save').disabled = !dirty;
  });
  $('mem-ed-save').onclick = () => {
    const text = memText.value.trim(); if (!text) return;
    $('mem-ed-status').textContent = 'сохраняю…';
    if (memSel === 'new') send({ type: 'memory_add', text }); else { memText._orig = memText.value; send({ type: 'memory_update', id: memSel, text }); }
  };
  $('mem-ed-del').onclick = () => { if (typeof memSel === 'number' && confirm('Удалить это воспоминание?')) send({ type: 'memory_delete', id: memSel }); };
  $('mem-ed-cancel').onclick = memClose;
  $('mem-new-btn').onclick = memNew;
  memItemsEl.addEventListener('click', (e) => { const item = e.target.closest('.hist-item'); if (!item) return; const it = memItems.find((x) => x.id === Number(item.dataset.id)); if (it) memOpen(it); });
  $('mem-search-btn').onclick = () => { memQuery = $('mem-search').value.trim(); memRefresh(); };
  $('mem-search').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); $('mem-search-btn').click(); } });
  $('mem-all').onclick = () => { memQuery = ''; $('mem-search').value = ''; memRefresh(); };

  /* ---------------------------------------------------------------- chat history sidebar */
  const HIST_KEY = 'nemoagent-history-open';
  const histPane = $('history'), histList = $('hist-list');
  let currentChatId = null, sideUserHidden = false;
  try { sideUserHidden = localStorage.getItem(HIST_KEY) === '0'; } catch (e) { /* ignore */ }
  // the sidebar exists only for the chat tab (chats) and the memory tab (memories)
  function updateSidebar() {
    const tab = activeTab();
    const wants = tab === 'chat' || tab === 'memory';
    histPane.classList.toggle('collapsed', !wants || sideUserHidden);   // collapsed keeps its column: the middle never jumps
    $('side-chats').classList.toggle('hidden', tab !== 'chat');
    $('side-mem').classList.toggle('hidden', tab !== 'memory');
    $('btn-history').classList.toggle('hidden', !wants);
  }
  $('btn-history').onclick = () => {
    sideUserHidden = !sideUserHidden;
    try { localStorage.setItem(HIST_KEY, sideUserHidden ? '0' : '1'); } catch (e) { /* ignore */ }
    updateSidebar();
  };
  updateSidebar();
  $('hist-new').onclick = () => send({ type: 'new_session' });
  $('hist-clear').onclick = () => { if (confirm('Очистить текущий чат? Он исчезнет из списка, а его записи будут стёрты из долговременной памяти.')) send({ type: 'clear_chat' }); };
  $('mem-prune').onclick = () => { $('mem-status').textContent = 'прибираюсь…'; send({ type: 'memory_prune' }); };
  $('mem-clear').onclick = () => { if (confirm('Стереть ВСЮ долговременную память на этом компьютере? Это необратимо.')) { $('mem-status').textContent = 'очищаю…'; send({ type: 'memory_clear' }); } };
  function fmtDate(ts) {
    const d = new Date(ts * 1000), now = new Date();
    const sameDay = d.toDateString() === now.toDateString();
    return sameDay ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : d.toLocaleDateString([], { day: '2-digit', month: '2-digit' }) + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  function renderChats(m) {
    currentChatId = m.current || null;
    const items = m.items || [];
    histList.innerHTML = items.length ? '' : '<div id="hist-empty">Пока пусто: первый диалог появится здесь после первого сообщения.</div>';
    for (const c of items) {
      const el = div('hist-item' + (c.id === currentChatId ? ' active' : ''));
      el.dataset.id = c.id;
      el.innerHTML = `<div class="t">${esc(c.title)}</div><div class="d">${fmtDate(c.updated)} · ${c.count} сообщ.</div><button class="x" title="удалить чат">✕</button>`;
      histList.appendChild(el);
    }
  }
  histList.addEventListener('click', (e) => {
    const x = e.target.closest('.x');
    const item = e.target.closest('.hist-item'); if (!item) return;
    if (x) { if (confirm('Удалить этот чат? Его записи будут стёрты и из долговременной памяти.')) send({ type: 'delete_chat', id: item.dataset.id }); return; }
    if (item.dataset.id !== currentChatId) send({ type: 'open_chat', id: item.dataset.id });
  });
  function renderHistory(c) {
    chat.innerHTML = ''; current = null; reasoningCard = null; execCard = null; waitingSince(null); $('stt-state').textContent = '';
    for (const e of c.messages || []) {
      if (e.role === 'user') {
        const d = div('msg user'); d._text = e.text || '';
        d.innerHTML = `<div class="src">${e.source === 'voice' ? '🎙 голос' : '⌨ текст'}${spkBtn(false, 'озвучить это сообщение')}</div>${fmt(e.text || '')}` + (e.attachments ? `<div class="src">📎 ${e.attachments} влож.</div>` : '');
        add(d);
      } else if (e.role === 'assistant') {
        const d = div('msg assistant'); d._text = e.text || ''; d._spoken = e.spoken ? (e.text || '') : '';
        d.innerHTML = spkBtn(!!e.spoken) + fmt(e.text || '') + (e.display ? `<div class="display">${fmt(e.display)}</div>` : '');
        add(d);
      } else if (e.role === 'task') card('task', '🎯 <b>задача исполнителю</b>', e.text || '', false);
      else if (e.role === 'report') card('report', '📋 <b>отчёт исполнителя</b>', e.text || '', false);
      else if (e.role === 'notice') add(div('notice', esc(e.text || '')));
    }
    document.querySelector('nav.tabs .tab[data-tab="chat"]').click();
  }
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
  window.addEventListener('error', (e) => { $('stt-state').textContent = '⚠ ошибка страницы: ' + (e.message || e.type); });
})();
