'use strict';
(() => {
  const $ = id => document.getElementById(id);
  let state = null;
  let sending = false;
  let connectionFailed = false;
  const dirty = new Set();
  const stages = {
    idle: 'Idle', resolving: 'Resolving model', downloading: 'Downloading', compiling_warp: 'Compiling warp',
    compiling: 'Compiling model', done: 'Done', failed: 'Failed', cancelled: 'Cancelled', waiting_egpu: 'Waiting for eGPU',
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
      const suffix = row.compiled ? ' (compiled)' : row.downloaded ? ' (downloaded)' : '';
      select.add(new Option(row.name + suffix, row.ref));
    });
    select.value = [...select.options].some(option => option.value === previous) ? previous : '';
  }

  function render() {
    const editable = !!state && state.offroad && !sending;
    document.querySelectorAll('button, select, input').forEach(element => { element.disabled = !editable; });
    if (!state) return;
    $('notice').textContent = state.offroad ? 'Offroad: eGPU management available' : 'Onroad: read-only, changes disabled';
    if (sending) return;
    $('enabled').checked = state.enabled;
    $('enabled').disabled = !editable || (!state.enabled && state.jetlink_enabled);
    fill('model', state.models, state.active.ref || '', 'No big model');
    fill('delete-model', state.models.filter(row => row.compiled || row.downloaded), '', 'Select model');
    $('select').disabled = !editable || state.busy;
    $('model').disabled = !editable || state.busy;
    $('cancel').disabled = !editable || !state.pending;
    $('delete').disabled = !editable || state.busy || !$('delete-model').value;
    $('clear').disabled = !editable || state.busy;
    const progress = state.progress;
    $('status').textContent = [
      `eGPU: ${state.present ? 'Connected' : 'Disconnected'} / ${state.enabled ? 'Enabled' : 'Disabled'}`,
      `This drive: ${state.failed ? 'Failed, using small model' : state.running ? 'Big model running' : state.loading ? 'Loading' : 'Not running'}`,
      `Selected model: ${state.active.display_name || state.active.short_name || 'None'}${state.ready ? ' (ready)' : ''}`,
      `Status: ${stages[progress.state] || progress.state || 'Idle'}${state.pending ? ' / Model request pending' : ''}`,
      typeof progress.progress === 'number' ? `Progress: ${progress.progress}%` : '',
      typeof progress.elapsed === 'number' ? `Elapsed: ${progress.elapsed} seconds` : '',
      progress.message || progress.detail || '',
      progress.error ? `Error: ${progress.error}` : '',
      state.jetlink_enabled ? 'Jetlink is enabled. Disable it in settings before enabling eGPU.' : '',
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
      $('notice').textContent = 'Unable to read device status. Controls disabled.';
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
      $('message').textContent = 'Request sent. Background status updates automatically.';
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
    if (window.confirm('Delete downloaded and compiled files for the selected model?')) act('delete', $('delete-model').value);
  });
  $('clear').addEventListener('click', () => {
    if (window.confirm('Clear the model cache? The active model will be kept.')) act('clear');
  });
  async function loop() {
    await poll();
    window.setTimeout(loop, 2000);
  }
  loop();
})();
