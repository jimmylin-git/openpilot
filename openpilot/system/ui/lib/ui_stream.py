"""MJPEG stream server for sunnypilot UI with telemetry overlay.
Imported lazily when STREAM=1."""
import threading
import io
import os
import json
import queue
import time
import base64
import hashlib
import struct
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

from PIL import Image
import pyray as rl


class StreamState:
    def __init__(self):
        self.frame = b""
        self.lock = threading.Lock()
        self.event = threading.Event()

    def update(self, jpeg):
        with self.lock:
            self.frame = jpeg
        self.event.set()

    def get(self):
        with self.lock:
            return self.frame

    def wait(self, t=2.0):
        self.event.wait(t)
        self.event.clear()


_CTRL_HTML = '<!DOCTYPE html><html><head>\n<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no,maximum-scale=1">\n<title>openpilot remote</title>\n<style>\n*{margin:0;padding:0;box-sizing:border-box;user-select:none;-webkit-user-select:none;touch-action:none}\nhtml,body{background:#000;height:100%;width:100%;overflow:hidden}\n#cam{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}\n#badge{position:absolute;top:8px;right:8px;z-index:9;background:rgba(0,0,0,0.55);color:#0f0;font:600 13px/1.4 -apple-system,sans-serif;padding:5px 10px;border-radius:8px;border:1px solid rgba(255,255,255,0.25);pointer-events:none}\n#toggle{position:absolute;bottom:10px;right:10px;z-index:9;background:rgba(0,0,0,0.6);color:#fff;font:600 13px -apple-system,sans-serif;padding:8px 14px;border-radius:10px;border:1px solid rgba(255,255,255,0.35)}\n#toggle.off{color:#888}\n</style></head><body>\n<img id="cam" src="/stream" draggable="false">\n<div id="badge">CTRL: OFF</div>\n<div id="toggle" ontouchstart="event.stopPropagation()" onclick="toggleCtrl(event)">ENABLE TOUCH</div>\n<script>\nlet ws=null, enabled=false, active={};\nfunction toggleCtrl(e){e.stopPropagation();enabled=!enabled;document.getElementById(\'toggle\').textContent=enabled?\'DISABLE TOUCH\':\'ENABLE TOUCH\';document.getElementById(\'toggle\').className=enabled?\'\':\'off\';document.getElementById(\'badge\').textContent=\'CTRL: \'+(enabled?\'ON\':\'OFF\');connect();}\nfunction connect(){\n  if(ws) try{ws.close()}catch(e){}\n  if(!enabled) return;\n  ws=new WebSocket((location.protocol===\'https:\'?\'wss://\':\'ws://\')+location.host+\'/ws\');\n  ws.onopen=()=>{document.getElementById(\'badge\').style.color=\'#0f0\'};\n  ws.onclose=()=>{document.getElementById(\'badge\').style.color=\'#f44\';if(enabled)setTimeout(connect,1000)};\n  ws.onerror=()=>{try{ws.close()}catch(e){}};\n}\nfunction norm(e){\n  const img=document.getElementById(\'cam\'), r=img.getBoundingClientRect();\n  const nw=img.naturalWidth, nh=img.naturalHeight;\n  if(!nw||!nh) return null;\n  let dw=r.width, dh=r.height, ox=0, oy=0;\n  if(dw/dh > nw/nh){ const h=dh; dw=dh*(nw/nh); ox=(r.width-dw)/2; }\n  else { const w=dw; dh=dw/(nw/nh); oy=(r.height-dh)/2; }\n  const x=(e.clientX-r.left-ox)/dw, y=(e.clientY-r.top-oy)/dh;\n  if(x<-0.02||x>1.02||y<-0.02||y>1.02) return null;\n  return {x:Math.min(1,Math.max(0,x)), y:Math.min(1,Math.max(0,y))};\n}\nfunction slotFor(id){ if(active[id]!==undefined) return active[id];\n  const used=Object.values(active); let slot=used.indexOf(0)<0?0:(used.indexOf(1)<0?1:0); active[id]=slot; return slot; }\nfunction send(t,e){ if(!ws||ws.readyState!==1) return; const n=norm(e); if(!n) return;\n  const slot=slotFor(e.pointerId!==undefined?e.pointerId:0);\n  ws.send(JSON.stringify({t:t,x:n.x,y:n.y,i:slot})); }\ndocument.addEventListener(\'pointerdown\',e=>{ if(!enabled) return; if(e.target.id===\'toggle\') return;\n  try{e.target.setPointerCapture&&e.target.setPointerCapture(e.pointerId)}catch(_){}\n  send(\'down\',e); });\ndocument.addEventListener(\'pointermove\',e=>{ if(!enabled) return; send(\'move\',e); });\nfunction end(e){ if(!enabled) return; send(\'up\',e); if(active[e.pointerId]!==undefined) delete active[e.pointerId]; }\ndocument.addEventListener(\'pointerup\',end);\ndocument.addEventListener(\'pointercancel\',end);\ndocument.addEventListener(\'contextmenu\',e=>e.preventDefault());\n</script></body></html>'

_OVERLAY_HTML = """<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>openpilot live</title>
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;overflow:hidden;height:100vh;width:100vw;margin:0;font-family:-apple-system,sans-serif}
#wrap{position:relative;width:100vw;height:100vh;display:flex;justify-content:center;align-items:center}
#cam{width:82%;height:95%;object-fit:contain;margin-left:auto;margin-right:5%}

/* Overlay container - matches image bounds */
#hud{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}

/* Speed cluster - center bottom */


/* Set speed - top right */
#set-speed{position:absolute;top:18%;left:1%;background:rgba(0,0,0,0.5);border-radius:12px;padding:6px 14px;text-align:center;border:1px solid rgba(255,255,255,0.15)}
#set-label{font-size:min(2.5vw,11px);color:#888;text-transform:uppercase;letter-spacing:1px}
#set-val{font-size:min(7vw,36px);font-weight:600;color:#fff}

/* Lead car info - center top */
#lead-info{position:absolute;top:38%;left:50%;transform:translateX(-50%);text-align:center;opacity:0;transition:opacity 0.3s;background:rgba(0,0,0,0.55);padding:6px 14px;border-radius:8px}
#lead-info.show{opacity:1}
#lead-dist{font-size:min(4.5vw,22px);font-weight:600;color:#fff;text-shadow:0 1px 4px rgba(0,0,0,0.8)}
#lead-gap{font-size:min(3.5vw,16px);font-weight:500;color:#4fc3f7}

/* Status bar - top left */
#status{position:absolute;top:5%;left:1%}
#engage-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:min(3vw,13px);font-weight:600;letter-spacing:1px;text-transform:uppercase}
#engage-badge.off{background:rgba(100,100,100,0.5);color:#888}
#engage-badge.on{background:rgba(76,175,80,0.3);color:#4caf50;border:1px solid rgba(76,175,80,0.4)}

/* Metrics strip - bottom left/right */
.metric{position:absolute;bottom:4%;font-size:min(3vw,13px);color:#aaa;text-shadow:0 1px 3px rgba(0,0,0,0.8)}
.metric .val{font-size:min(4.5vw,20px);font-weight:600;color:#e0e0e0}
#m-steer{left:4%}
#m-grade{left:1%}
#m-accel{position:absolute;left:50%;top:10%;transform:translateX(-50%);text-align:center;width:min(50vw,220px)}
#accel-label{font-size:min(2.5vw,10px);color:#888;letter-spacing:1px;margin-bottom:2px}
#accel-bar-wrap{display:flex;align-items:center;height:min(2.5vw,12px);background:rgba(255,255,255,0.1);border-radius:6px;overflow:hidden;position:relative}
#accel-bar-neg{height:100%;width:0;background:#f44336;position:absolute;right:50%;border-radius:6px 0 0 6px;transition:width 0.1s}
#accel-bar-pos{height:100%;width:0;background:#4caf50;position:absolute;left:50%;border-radius:0 6px 6px 0;transition:width 0.1s}
#accel-center{position:absolute;left:50%;top:0;bottom:0;width:2px;background:rgba(255,255,255,0.4);transform:translateX(-50%);z-index:1}
#accel-num{font-size:min(4vw,18px);color:#aaa;margin-top:1px}

/* Brake/Gas indicators */
#pedals{position:absolute;bottom:15%;left:1%;display:flex;gap:10px;align-items:flex-end}
.pedal-wrap{display:flex;flex-direction:column;align-items:center;gap:2px}
.pedal-label{font-size:min(2.5vw,10px);color:#888;letter-spacing:1px}
.pedal-bar{width:min(4vw,18px);min-height:2px;border-radius:3px;transition:height 0.15s}
.pedal-val{font-size:min(2.5vw,11px);color:#aaa}
#gas-bar{background:#4caf50}
#brake-bar{background:#f44336}
#perf-strip{position:absolute;bottom:1%;left:50%;transform:translateX(-50%);display:flex;gap:min(3vw,14px);background:rgba(0,0,0,0.5);padding:3px 10px;border-radius:6px}
.pf{text-align:center}
.pf-label{font-size:min(1.8vw,8px);color:#666;letter-spacing:0.5px}
.pf-val{font-size:min(2.5vw,12px);color:#e0e0e0;font-weight:500}
.pf-val.bad{color:#f44336}

</style></head><body>
<div id="wrap">
  <img id="cam" src="/stream">
  <div id="hud"><div ontouchend="event.preventDefault();event.stopPropagation();toggleFS();" onclick="toggleFS()" style="position:absolute;right:1%;top:5%;background:rgba(0,0,0,0.4);border:1px solid rgba(255,255,255,0.2);color:rgba(255,255,255,0.6);font-size:16px;padding:8px 12px;border-radius:8px;pointer-events:auto;z-index:9999;cursor:pointer">&#x26F6;</div>
    <div id="status"><span id="engage-badge" class="off">OFF</span></div>

    <div id="set-speed">
      <div id="set-label">SET</div>
      <div id="set-val">--</div>
    </div>

    <div id="lead-info">
      <div id="lead-dist">--</div>
      <div id="lead-gap">--</div>
    </div>



    
    
    <div id="m-accel">
      <div id="accel-label">ACCEL</div>
      <div id="accel-bar-wrap">
        <div id="accel-bar-neg" class="accel-fill"></div>
        <div id="accel-center"></div>
        <div id="accel-bar-pos" class="accel-fill"></div>
      </div>
      <div id="accel-num">0.0</div>
    </div>

    <div class="metric" id="m-grade"><div class="val" id="grade-val">--%</div>grade</div>
    <div id="pedals">
      <div class="pedal-wrap"><div class="pedal-label">GAS</div><div id="gas-bar" class="pedal-bar"></div><div class="pedal-val" id="gas-val">0</div></div>
      <div class="pedal-wrap"><div class="pedal-label">BRK</div><div id="brake-bar" class="pedal-bar"></div><div class="pedal-val" id="brake-val">0</div></div>
    </div>
    <div id="perf-strip">
      <div class="pf"><div class="pf-label">MODEL</div><div class="pf-val" id="pf-model">--</div></div>
      <div class="pf"><div class="pf-label">DROPS</div><div class="pf-val" id="pf-drops">--</div></div>
      <div class="pf"><div class="pf-label">CPU</div><div class="pf-val" id="pf-cpu">--</div></div>
      <div class="pf"><div class="pf-label">MEM</div><div class="pf-val" id="pf-mem">--</div></div>
      <div class="pf"><div class="pf-label">CPU TEMP</div><div class="pf-val" id="pf-temp">--</div></div>
    </div>
  </div>
</div>
<script>
let lastData = null;
function poll() {
  fetch('/telemetry').then(r => r.json()).then(d => {
    lastData = d;
    // Speed


    // Set speed
    const sv = document.getElementById('set-val');
    sv.textContent = d.setSpeed > 0 ? d.setSpeed : '--';

    // Engage status
    const badge = document.getElementById('engage-badge');
    const engaged = d.cruiseEnabled === true || d.driveState === 'active';
    const standby = !engaged && (d.driveState === 'standby' || d.cruiseEnabled === false);
    badge.className = engaged ? 'on' : 'off';
    badge.textContent = engaged ? 'ENGAGED' : standby ? 'STANDBY' : 'OFF';

    // Lead car
    const li = document.getElementById('lead-info');
    const isEngaged = d.cruiseEnabled === true || d.driveState === 'active';
    if (isEngaged && d.leadDist !== undefined && d.leadDist !== null) {
      li.className = 'show';
      const ft = Math.round(d.leadDist * 3.28084);
      const egoMph = d.vEgo * 2.23694;
      const gap = d.vEgo > 0.5 ? (d.leadDist / d.vEgo).toFixed(1) : '--';
      document.getElementById('lead-dist').textContent = ft + ' ft';
      document.getElementById('lead-gap').textContent = gap + ' s';
    } else {
      li.className = '';
    }

    // Steer
    

    // Grade
    
    

    // Accel
    const a = d.aEgo || 0;
    const pct = Math.min(Math.abs(a) / 3.0 * 50, 50);
    document.getElementById('accel-bar-pos').style.width = (a > 0 ? pct : 0) + '%';
    document.getElementById('accel-bar-neg').style.width = (a < 0 ? pct : 0) + '%';
    document.getElementById('accel-num').textContent = a.toFixed(1) + ' m/s2';

    // CPU
    if (d.cpuTemp !== undefined) { var e=document.getElementById("pf-temp"); e.textContent=d.cpuTemp+String.fromCharCode(176); e.className="pf-val"+(d.cpuTemp>80?" bad":""); }

    // Gas/Brake bars
    if (d.grade !== undefined) document.getElementById('grade-val').textContent = d.grade + '%';
    document.getElementById('gas-bar').style.height = Math.max(2, d.gas * 0.6) + 'px';
    document.getElementById('gas-val').textContent = d.gas;
    document.getElementById('brake-bar').style.height = Math.max(2, d.brake * 0.6) + 'px';
    document.getElementById('brake-val').textContent = d.brake;
    if (d.modelExec !== undefined) { var e=document.getElementById("pf-model"); e.textContent=d.modelExec+"ms"; e.className="pf-val"+(d.modelExec>35?" bad":""); }
    if (d.frameDropPerc !== undefined) { var e=document.getElementById("pf-drops"); e.textContent=d.frameDropPerc+"%"; e.className="pf-val"+(d.frameDropPerc>5?" bad":""); }
    if (d.cpuUsage !== undefined) { var e=document.getElementById("pf-cpu"); e.textContent=d.cpuUsage+"%"; e.className="pf-val"+(d.cpuUsage>85?" bad":""); }
    if (d.memUsed !== undefined) { var e=document.getElementById("pf-mem"); e.textContent=d.memUsed+"%"; e.className="pf-val"+(d.memUsed>85?" bad":""); }

  }).catch(() => {});
  setTimeout(poll, 250);
}
document.addEventListener("DOMContentLoaded",function(){
  setTimeout(function(){window.scrollTo(0,1);},100);
  setTimeout(function(){window.scrollTo(0,0);},200);
});
function toggleFS(){var d=document.documentElement;try{if(!document.fullscreenElement&&!document.webkitFullscreenElement){if(d.requestFullscreen)d.requestFullscreen();else if(d.webkitRequestFullscreen)d.webkitRequestFullscreen(Element.ALLOW_KEYBOARD_INPUT);else if(d.webkitEnterFullscreen)d.webkitEnterFullscreen();else alert('Fullscreen not supported');}else{if(document.exitFullscreen)document.exitFullscreen();else if(document.webkitExitFullscreen)document.webkitExitFullscreen();}}catch(e){alert('FS error: '+e);}}
poll();
</script></body></html>"""


class StreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path == "/ws":
            _handle_ws(self)
        elif self.path == "/ctrl":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_CTRL_HTML)))
            self.end_headers()
            self.wfile.write(_CTRL_HTML.encode())
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    self.server._state.wait(2.0)
                    f = self.server._state.get()
                    if f:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(f)).encode() + b"\r\n\r\n")
                        self.wfile.write(f)
                        self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/snapshot":
            f = self.server._state.get()
            if f:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(f)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(f)
            else:
                self.send_response(503)
                self.end_headers()
        elif self.path == "/telemetry":
            try:
                with open("/tmp/telemetry.json", "r") as f:
                    data = f.read().encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            except:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_OVERLAY_HTML)))
            self.end_headers()
            self.wfile.write(_OVERLAY_HTML.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


_state = None
_counter = 0

_state = None
_counter = 0
_last_tel = 0.0
_enc_queue = None
_app = None
_slot_pressed = [False, False]
_inject_logged = False



def _ws_accept(key):
  guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
  return base64.b64encode(hashlib.sha1((key + guid).encode()).digest()).decode()

def _ws_read_frame(rfile):
  hdr = rfile.read(2)
  if len(hdr) < 2:
    return None, None
  b0, b1 = hdr[0], hdr[1]
  opcode = b0 & 0x0F
  ln = b1 & 0x7F
  if ln == 126:
    e = rfile.read(2)
    if len(e) < 2: return None, None
    ln = struct.unpack(">H", e)[0]
  elif ln == 127:
    e = rfile.read(8)
    if len(e) < 8: return None, None
    ln = struct.unpack(">Q", e)[0]
  mask = rfile.read(4) if (b1 & 0x80) else b""
  payload = rfile.read(ln) if ln else b""
  if mask and len(mask) == 4:
    payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
  return opcode, payload

def _ws_send_frame(wfile, opcode, payload=b""):
  hdr = bytes([0x80 | opcode])
  ln = len(payload)
  if ln < 126:
    hdr += bytes([ln])
  elif ln < 65536:
    hdr += bytes([126]) + struct.pack(">H", ln)
  else:
    hdr += bytes([127]) + struct.pack(">Q", ln)
  wfile.write(hdr + payload)
  wfile.flush()

def _inject_event(kind, x, y, slot):
  global _app, _slot_pressed
  app = _app
  if app is None:
    return
  try:
    from openpilot.system.ui.lib.application import MouseEvent, MousePos
    mouse = app._mouse
    w = float(app.width)
    h = float(app.height)
    px = max(0.0, min(w, float(x) * w))
    py = max(0.0, min(h, float(y) * h))
    now = time.monotonic()
    slot = 0 if int(slot) % 2 == 0 else 1
    pressed = _slot_pressed[slot]
    if kind == "down":
      if pressed: return
      ev = MouseEvent(MousePos(px, py), slot, True, False, True, now)
      _slot_pressed[slot] = True
    elif kind == "move":
      if not pressed: return
      ev = MouseEvent(MousePos(px, py), slot, False, False, True, now)
    elif kind == "up":
      if not pressed: return
      ev = MouseEvent(MousePos(px, py), slot, False, True, False, now)
      _slot_pressed[slot] = False
    else:
      return
    with mouse._lock:
      mouse._events.append(ev)
    global _inject_logged
    if not _inject_logged:
      _inject_logged = True
      try:
        from openpilot.common.swaglog import cloudlog
        cloudlog.warning("remote UI input active (first event)")
      except Exception:
        pass
  except Exception as e:
    try:
      from openpilot.common.swaglog import cloudlog
      cloudlog.error("remote input error: " + str(e))
    except Exception:
      pass

def _handle_ws(handler):
  try:
    key = handler.headers.get("Sec-WebSocket-Key", "")
    if not key:
      return
    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", _ws_accept(key))
    handler.end_headers()
    handler.wfile.flush()
    while True:
      opcode, payload = _ws_read_frame(handler.rfile)
      if opcode is None:
        break
      if opcode == 0x8:
        break
      if opcode == 0x9:
        _ws_send_frame(handler.wfile, 0xA, payload)
        continue
      if opcode == 0x1:
        try:
          msg = json.loads(payload.decode("utf-8", "replace"))
          t = msg.get("t")
          if t in ("down", "move", "up"):
            _inject_event(t, float(msg.get("x", 0.0)), float(msg.get("y", 0.0)), int(msg.get("i", 0)))
        except Exception:
          pass
  except Exception:
    pass

def _encode_worker():
  while True:
    item = _enc_queue.get()
    if item is None:
      break
    raw, w, h, quality = item
    try:
      img = Image.frombytes("RGBA", (w, h), raw).transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")
      buf = io.BytesIO()
      img.save(buf, "JPEG", quality=quality)
      if _state is not None:
        _state.update(buf.getvalue())
    except Exception:
      pass


def start(port=8082):
  """Start the MJPEG HTTP server + JPEG encode worker in background threads."""
  global _state, _enc_queue
  _state = StreamState()
  _enc_queue = queue.Queue(maxsize=3)
  threading.Thread(target=_encode_worker, daemon=True).start()
  srv = ThreadingHTTPServer(("0.0.0.0", port), StreamHandler)
  srv._state = _state
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  return _state


def _write_telemetry():
  global _last_tel
  now = time.time()
  if now - _last_tel < 0.25:
    return
  _last_tel = now
  try:
    from openpilot.selfdrive.ui.ui_state import ui_state as us
    sm = us.sm
  except Exception:
    return
  data = {}
  try:
    cs = sm["carState"]
    data["vEgo"] = round(float(cs.vEgo), 2)
    data["aEgo"] = round(float(cs.aEgo), 2)
    data["gas"] = 100 if cs.gasPressed else 0
    data["brake"] = 100 if cs.brakePressed else 0
  except Exception:
    pass
  try:
    ss = sm["selfdriveState"]
    data["cruiseEnabled"] = bool(ss.enabled)
    data["driveState"] = "active" if ss.enabled else ("standby" if getattr(us, "started", False) else "off")
    vc = float(ss.vCruiseCluster if ss.vCruiseCluster > 0 else ss.vCruise)
    if vc > 0:
      data["setSpeed"] = int(round(vc * 2.23694))  # mph
  except Exception:
    pass
  try:
    mv = sm["modelV2"]
    if len(mv.lead) > 0:
      data["leadDist"] = round(float(mv.lead[0].dRel), 1)  # meters
    data["modelExec"] = int(round(float(mv.modelExecutionTime) * 1000.0))  # ms
  except Exception:
    pass
  try:
    dev = sm["deviceState"]
    temps = [float(t) for t in dev.cpuTempC]
    if temps:
      data["cpuTemp"] = int(round(max(temps)))
    cpus = [int(c) for c in dev.cpuUsagePercent]
    if cpus:
      data["cpuUsage"] = max(cpus)
    data["memUsed"] = int(dev.memoryUsagePercent)
  except Exception:
    pass
  try:
    with open("/tmp/telemetry.json", "w") as f:
      json.dump(data, f)
  except Exception:
    pass


def capture_frame(app, quality=50, target_fps=10):
  global _app
  if _app is None:
    _app = app
  """Call from the render loop: snapshot texture, queue raw for async JPEG encode."""
  global _counter
  if _state is None or app._render_texture is None:
    return
  _counter += 1
  skip = max(1, int(getattr(app, "_target_fps", 60)) // max(1, int(target_fps)))
  if _counter % skip != 0:
    return
  _write_telemetry()
  try:
    si = rl.load_image_from_texture(app._render_texture.texture)
    w, h = si.width, si.height
    raw = bytes(rl.ffi.buffer(si.data, w * h * 4))
    rl.unload_image(si)
    if _enc_queue is not None:
      try:
        _enc_queue.put_nowait((raw, w, h, int(quality)))
      except Exception:
        pass
  except Exception:
    pass
