"""Hardware-free execution of the production router and frame transport."""
import ast
import mmap
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
FLAGS = {f'P_{name.upper()}': name for name in ('enabled', 'present', 'failed', 'loading', 'running', 'ready')}


def definitions(filename, namespace):
  tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
  nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.Assign))]
  exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)
  return namespace


class FakeWorker:
  def __init__(self, *args):
    self.args = args
    self.ready = threading.Event()
    self.done = threading.Event()
    self.error = None
    self.run = Mock(return_value={'source': 'big'})
    self.close = Mock()


def setup_router():
  clock = SimpleNamespace(now=100.0)
  flags = dict(enabled=True, present=True, loading=False, running=False, ready=False, failed=False)
  params = SimpleNamespace(get_bool=lambda k: flags[k], put_bool=lambda k, v, **kw: flags.update({k: v}))
  small = SimpleNamespace(run=Mock(return_value={'source': 'small'}), vision_input_names=['img', 'big_img'])
  reset = Mock()
  factory = Mock(side_effect=FakeWorker)
  bundle = {'generation': 11, 'overrides': {'lat': '0.2', 'long': '0.5'}}
  namespace = definitions('dragonpilot/egpu/joining.py', {
    **FLAGS, 'atexit': SimpleNamespace(register=lambda cb: None), 'math': __import__('math'),
    'time': SimpleNamespace(monotonic=lambda: clock.now), 'Worker': FakeWorker, 'EgpuAction': object,
    'get_active_bundle': lambda p: bundle, 'active_ready': lambda b: b is not None,
  })
  logger = Mock()
  router = namespace['JoiningEgpu'](small, 1928, 1208, 8, params, reset, logger, 0.0, 0.3, factory, lambda: clock.now)
  inputs = {'action_t': np.array([0.1, 0.4], dtype=np.float32), 'desire_pulse': np.zeros(8)}
  def run():
    return router.run({}, {}, inputs)
  return SimpleNamespace(router=router, run=run, clock=clock, flags=flags, small=small,
                         reset=reset, factory=factory, inputs=inputs, bundle=bundle, logger=logger)


def activate(s):
  for _ in range(3):
    assert s.run()['source'] == 'small'
  s.router.worker.ready.set()
  s.router.in_control = False
  assert s.run()['source'] == 'big'
  assert s.flags['running'] and not s.flags['loading'] and not s.flags['ready']


def test_small_runs_during_load_and_indefinite_control_wait():
  s = setup_router()
  assert s.run()['source'] == 'small'
  assert s.flags['loading'] and not s.flags['running']
  s.router.worker.ready.set()
  for _ in range(100):
    s.clock.now += 10
    assert s.run()['source'] == 'small'
    assert s.flags['ready'] and not s.flags['loading']
  assert s.factory.call_count == 1
  s.router.in_control = False
  assert s.run()['source'] == 'big'
  assert s.router.handovers == 1
  np.testing.assert_allclose(s.inputs['action_t'], [0.1, 0.4])
  np.testing.assert_allclose(s.router.worker.run.call_args.args[2]['action_t'], [0.3, 0.6])


def test_three_small_outputs_required_before_switch():
  s = setup_router()
  s.run()
  s.router.worker.ready.set()
  s.router.in_control = False
  assert s.run()['source'] == 'small'
  assert s.run()['source'] == 'small'
  assert s.run()['source'] == 'big'


@pytest.mark.parametrize('failure', ['timeout', 'disconnect', 'disabled', 'watchdog', 'drops', 'exit'])
def test_failure_resets_small_and_retries_only_after_cooldown_and_old_worker_exit(failure):
  s = setup_router()
  activate(s)
  worker = s.router.worker
  if failure == 'timeout':
    worker.run.side_effect = TimeoutError('80ms')
  elif failure == 'disconnect':
    s.flags['present'] = False
  elif failure == 'disabled':
    s.flags['enabled'] = False
  elif failure == 'watchdog':
    s.flags['failed'] = True
  elif failure == 'drops':
    s.router.frame_drop_ratio = 0.008
  else:
    worker.done.set()
  assert s.run()['source'] == 'small'
  assert s.flags['failed'] and not s.flags['running'] and not s.flags['ready']
  s.reset.assert_called_once()
  s.logger.error.assert_called_once()
  worker.close.assert_called_once()
  s.flags.update(present=True, enabled=True)
  s.router.frame_drop_ratio = 0
  s.router.in_control = True
  s.clock.now = 104.999
  s.run()
  assert s.factory.call_count == 1
  s.clock.now = 105
  if failure != 'exit':
    s.run()
    assert s.factory.call_count == 1  # Previous process has not been reaped.
  worker.done.set()
  assert s.run()['source'] == 'small'
  assert s.factory.call_count == 2
  assert s.flags['loading'] and not s.flags['failed']
  s.router.worker.ready.set()
  assert s.run()['source'] == 'small'  # Retry is allowed during control, switching is not.
  s.router.in_control = False
  assert s.run()['source'] == 'big'


def test_invalid_bundle_overrides_log_failure_without_worker():
  s = setup_router()
  s.bundle['overrides']['lat'] = 'nan'
  assert s.run()['source'] == 'small'
  assert s.flags['failed']
  s.factory.assert_not_called()
  s.logger.error.assert_called_once()


def test_disabled_start_does_not_load_until_enabled():
  s = setup_router()
  s.flags['enabled'] = False
  assert s.run()['source'] == 'small'
  s.factory.assert_not_called()
  s.flags['enabled'] = True
  s.run()
  s.factory.assert_called_once()


def test_production_wiring_and_ready_key_registration():
  tree = ast.parse((ROOT / 'selfdrive/modeld/modeld.py').read_text(encoding='utf-8'))
  main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
  wrapper = next(n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'JoiningEgpu')
  assert ast.unparse(wrapper.args[0]) == 'small_model'
  control = next(n for n in ast.walk(main) if isinstance(n, ast.Assign) and any(
    isinstance(t, ast.Attribute) and t.attr == 'in_control' for t in n.targets))
  assert ast.unparse(control.value) == 'jetlink_adapter.in_control(sm)'
  dispatch = next(n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == 'EGPU or EGPU_JOIN')
  assert ast.unparse(dispatch.body[-1].value) == 'model.run(bufs, transforms, inputs)'
  action = next(n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == 'EGPU or (EGPU_JOIN and model.chestnut)')
  assert 'model.get_action' in ast.unparse(action.body[0])
  keys = (ROOT / 'common/params_keys.h').read_text(encoding='utf-8')
  assert '{"dp_egpu_ready", {CLEAR_ON_MANAGER_START|CLEAR_ON_OFFROAD_TRANSITION, BOOL, "0"}}' in keys
  ns = {}
  exec((ROOT / 'dragonpilot/settings/min-feat.dev.egpu.py').read_text(encoding='utf-8'), ns)
  ready = next(item for item in ns['ITEMS'] if item['key'] == 'dp_egpu_ready')
  assert ready['flags'] == 'CLEAR_ON_MANAGER_START|CLEAR_ON_OFFROAD_TRANSITION'


@pytest.mark.parametrize('flags,expected', [
  ({'loading': True}, 'LOADING'),
  ({'ready': True}, 'READY'),
  ({'running': True}, 'ACTIVE'),
  ({'failed': True}, 'FAILED'),
  ({}, 'NO_MODEL'),
])
def test_main_ui_matches_background_worker_state(flags, expected):
  tree = ast.parse((ROOT / 'selfdrive/ui/ui_state.py').read_text(encoding='utf-8'))
  method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_update_egpu')
  constants = {**FLAGS, 'P_ACTIVE_BUNDLE': 'bundle', 'P_STATUS': 'status'}
  states = SimpleNamespace(**{s: s for s in ('DISCONNECTED', 'FAILED', 'LOADING', 'ACTIVE', 'READY', 'NO_MODEL')})
  values = {'present': True, 'enabled': True, 'bundle': {'ref': 'model'}, **flags}
  obj = SimpleNamespace(params=SimpleNamespace(get_bool=lambda k: bool(values.get(k)), get=lambda k: values.get(k)),
                        started=True, usbgpu=False, usbgpu_compiled=False)
  ns = {'egpu_const': SimpleNamespace(**constants), 'EgpuState': states}
  exec(compile(ast.Module(body=[method], type_ignores=[]), '<egpu ui state>', 'exec'), ns)
  ns['_update_egpu'](obj)
  assert obj.egpu_state == expected


def ipc_namespace():
  return definitions('dragonpilot/egpu/ipc.py', {
    'FIRST_OUTPUT_TIMEOUT': 150, 'threading': threading, 'queue': queue,
    'np': np, 'os': __import__('os'), 'sys': __import__('sys'), 'Path': Path,
  })


def test_transport_copies_two_frames_and_preserves_inputs():
  ns = ipc_namespace()
  worker = ns['Worker'].__new__(ns['Worker'])
  worker.ready, worker.stopped = threading.Event(), threading.Event()
  worker.ready.set()
  worker.error = None
  worker.frame_size = 8
  worker.buffer = mmap.mmap(-1, 16)
  captured = []
  def request(req):
    transforms, inputs, result, event = req
    captured.append((transforms, inputs))
    result.append({'plan': 'parsed'})
    event.set()
  worker.requests = SimpleNamespace(put_nowait=request)
  try:
    bufs = {'img': np.arange(8, dtype=np.uint8), 'big_img': np.arange(8, 16, dtype=np.uint8)}
    assert worker.run(bufs, {}, {}) == {'plan': 'parsed'}
    assert worker.buffer[:] == bytes(range(16))
    assert captured == [({}, {})]
  finally:
    worker.buffer.close()


def test_transport_timeout_closes_worker_without_waiting_for_gpu():
  ns = ipc_namespace()
  worker = ns['Worker'].__new__(ns['Worker'])
  worker.ready, worker.stopped = threading.Event(), threading.Event()
  worker.ready.set()
  worker.error = None
  worker.frame_size = 1
  worker.buffer = bytearray(2)
  worker.requests = SimpleNamespace(put_nowait=lambda request: None)
  worker.process = Mock()
  worker.process.poll.return_value = None
  with pytest.raises(TimeoutError, match='80 ms'):
    worker.run({'img': b'a', 'big_img': b'b'}, {}, {})
  worker.process.kill.assert_called_once()
  worker.process.wait.assert_not_called()
  assert worker.stopped.is_set() and not worker.ready.is_set()


def test_worker_background_thread_is_not_fifo_or_on_core_seven():
  ns = ipc_namespace()
  os_mock = Mock()
  os_mock.SCHED_OTHER = 0
  os_mock.sched_param.return_value = 'normal'
  os_mock.cpu_count.return_value = 8
  ns['os'] = os_mock
  ns['background']()
  os_mock.sched_setscheduler.assert_called_once_with(0, 0, 'normal')
  os_mock.sched_setaffinity.assert_called_once_with(0, set(range(7)))


@pytest.mark.parametrize('failure', ['load_timeout', 'load_error', 'reply_error', None])
def test_ipc_thread_reports_failures_and_always_reaps_child_and_closes_fds(failure):
  ns = ipc_namespace()
  worker = ns['Worker'].__new__(ns['Worker'])
  worker.width, worker.height, worker.frame_size, worker.bundle = 10, 10, 8, {}
  worker.ready, worker.stopped, worker.done = threading.Event(), threading.Event(), threading.Event()
  worker.error = worker.process = worker.buffer = None
  parent, child, conn, proc = Mock(), Mock(), Mock(), Mock()
  proc.poll.return_value = None
  conn.poll.return_value = failure != 'load_timeout'
  conn.recv.side_effect = ([{'error': 'load failed'}] if failure == 'load_error'
                           else ['ready', {'error': 'bad inference'} if failure == 'reply_error' else {'plan': 'result'}])
  result, event = [], threading.Event()
  def get_request(**kwargs):
    worker.stopped.set()
    return {}, {}, result, event
  worker.requests = SimpleNamespace(get=get_request)
  os_mock = Mock()
  ns.update(background=lambda: None, socket=SimpleNamespace(socketpair=lambda: (parent, child)),
            Connection=lambda fd: conn, subprocess=SimpleNamespace(Popen=Mock(return_value=proc)),
            mmap=SimpleNamespace(mmap=lambda *args: bytearray(16)), os=os_mock,
            __file__=str(ROOT / 'dragonpilot/egpu/ipc.py'))
  worker._loop()
  assert worker.done.is_set() and not worker.ready.is_set()
  proc.kill.assert_called_once()
  proc.wait.assert_called_once()
  conn.close.assert_called_once()
  parent.close.assert_called_once()
  os_mock.close.assert_called_once()
  if failure == 'load_timeout':
    assert isinstance(worker.error, TimeoutError)
    conn.poll.assert_called_once_with(150.0)
  elif failure:
    assert isinstance(worker.error, RuntimeError)
  else:
    assert worker.error is None
    assert result == [{'plan': 'result'}]
    assert event.is_set()
