'use strict';
(() => {
  const $ = id => document.getElementById(id);
  let state = null;
  let sending = false;
  let connectionFailed = false;
  const dirty = new Set();
  const stages = {
    idle: '待命', resolving: '解析模型', downloading: '下載中', compiling_warp: '編譯影像處理',
    compiling: '編譯模型', done: '完成', failed: '失敗', cancelled: '已取消', waiting_egpu: '等待接上 eGPU',
  };

  async function api(options) {
    const response = await fetch('/api/egpu', {cache: 'no-store', ...options});
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
    return response.json();
  }

  function fill(id, rows, selected, emptyLabel) {
    const select = $(id);
    const previous = dirty.has(id) ? select.value : selected;
    select.replaceChildren(new Option(emptyLabel, ''));
    rows.forEach(row => {
      const suffix = row.compiled ? '（已編譯）' : row.downloaded ? '（已下載）' : '';
      select.add(new Option(row.name + suffix, row.ref));
    });
    select.value = [...select.options].some(option => option.value === previous) ? previous : '';
  }

  function render() {
    const editable = !!state && state.offroad && !sending;
    document.querySelectorAll('button, select, input').forEach(element => { element.disabled = !editable; });
    if (!state) return;
    $('notice').textContent = state.offroad ? 'Offroad：可以管理 eGPU' : 'Onroad：僅供查看，禁止變更';
    if (sending) return;
    $('enabled').checked = state.enabled;
    $('enabled').disabled = !editable || (!state.enabled && state.jetlink_enabled);
    fill('model', state.models, state.active.ref || '', '不使用大模型');
    fill('delete-model', state.models.filter(row => row.compiled || row.downloaded), '', '選擇模型');
    $('select').disabled = !editable || state.busy;
    $('model').disabled = !editable || state.busy;
    $('cancel').disabled = !editable || !state.pending;
    $('delete').disabled = !editable || state.busy || !$('delete-model').value;
    $('clear').disabled = !editable || state.busy;
    const progress = state.progress;
    $('status').textContent = [
      `eGPU：${state.present ? '已連接' : '未連接'} / ${state.enabled ? '啟用' : '停用'}`,
      `本趟：${state.failed ? '失敗，退回小模型' : state.running ? '大模型執行中' : state.loading ? '載入中' : '未執行'}`,
      `選定模型：${state.active.display_name || state.active.short_name || '無'}${state.ready ? '（已備妥）' : ''}`,
      `處理：${stages[progress.state] || progress.state || '待命'}${state.pending ? ' / 有模型處理要求' : ''}`,
      typeof progress.progress === 'number' ? `進度：${progress.progress}%` : '',
      typeof progress.elapsed === 'number' ? `已用時間：${progress.elapsed} 秒` : '',
      progress.message || progress.detail || '',
      progress.error ? `錯誤：${progress.error}` : '',
      state.jetlink_enabled ? 'Jetlink 已開啟；請先在既有設定中關閉，再啟用 eGPU。' : '',
    ].filter(Boolean).join('\n');
  }

  async function poll() {
    try {
      state = await api();
      if (connectionFailed) $('message').textContent = '';
      connectionFailed = false;
      render();
    } catch (error) {
      connectionFailed = true;
      state = null;
      render();
      $('notice').textContent = '無法讀取裝置狀態，已停用操作';
      $('message').textContent = error.message;
    }
  }

  async function act(action, value) {
    if (sending || !state || !state.offroad) return;
    sending = true;
    render();
    $('message').textContent = '';
    try {
      await api({method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action, value})});
      if (action === 'select') dirty.delete('model');
      if (action === 'delete') dirty.delete('delete-model');
      $('message').textContent = '已送出要求；背景處理狀態會自動更新。';
    } catch (error) {
      $('message').textContent = error.message;
    } finally {
      sending = false;
      await poll();
    }
  }

  ['model', 'delete-model'].forEach(id => $(id).addEventListener('change', () => { dirty.add(id); render(); }));
  $('enabled').addEventListener('change', () => act('enabled', $('enabled').checked));
  $('select').addEventListener('click', () => act('select', $('model').value));
  $('cancel').addEventListener('click', () => act('cancel'));
  $('refresh').addEventListener('click', () => act('refresh'));
  $('delete').addEventListener('click', () => {
    if (window.confirm('確定刪除選取模型的下載及編譯檔？')) act('delete', $('delete-model').value);
  });
  $('clear').addEventListener('click', () => {
    if (window.confirm('確定清除模型快取？目前使用中的模型會保留。')) act('clear');
  });
  async function loop() {
    await poll();
    window.setTimeout(loop, 2000);
  }
  loop();
})();
