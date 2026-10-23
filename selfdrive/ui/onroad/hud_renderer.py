import math
import shutil
import threading
import time
import pyray as rl
from dataclasses import dataclass
from cereal import custom
from openpilot.common.constants import CV
from openpilot.selfdrive.ui.onroad.exp_button import ExpButton
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.ui.ui_state import JetlinkState
from dragonpilot.selfdrive.ui import jetlink_ui
from dragonpilot.selfdrive.ui import egpu_ui
from openpilot.selfdrive.ui.ui_state import EgpuState
#from openpilot.selfdrive.ui.mici.onroad.torque_bar import TorqueBar

# Constants
SET_SPEED_NA = 255
KM_TO_MILE = 0.621371
CRUISE_DISABLED_CHAR = '–'
TRAFFIC_STOP_IDX = int(custom.LongitudinalPlanDP.LongitudinalPlanSource.trafficStop)

# dp - 紅綠燈減速中 banner（基準尺寸 x1.5）：中央車速正下方
TS_HOLD_TIME = 0.5          # 秒；braking 訊號短暫掉落時維持顯示，避免閃爍
TS_FONT_SIZE = 144
TS_BANNER_HEIGHT = 225
TS_TOP_OFFSET = 5           # 相對 header_height 的位移（km/h 單位文字底緣約在 header 內）
TS_PADDING_X = 60
TS_ICON_TEXT_GAP = 36
TS_LIGHT_RADIUS = 36
TS_LIGHT_GAP = 18
TS_BORDER = 9

# dp - jetlink: onroad badge under the experimental-mode button
JETLINK_BADGE_W = 120
JETLINK_BADGE_H = 89
JETLINK_LABEL_SIZE = 40

# DP Perf Constants (字體恢復為 50)
PERF_FONT_SIZE = 50
PERF_PADDING = 14
PERF_MARGIN_BOTTOM = 0   # 設定為 0 以貼齊底部
PERF_ITEM_GAP = 50
PERF_BG_COLOR = rl.Color(0, 0, 0, 160)


@dataclass(frozen=True)
class UIConfig:
  header_height: int = 300
  border_size: int = 30
  button_size: int = 192
  set_speed_width_metric: int = 200
  set_speed_width_imperial: int = 172
  set_speed_height: int = 204
  wheel_icon_size: int = 144


@dataclass(frozen=True)
class FontSizes:
  current_speed: int = 176
  speed_unit: int = 66
  max_speed: int = 40
  set_speed: int = 90


@dataclass(frozen=True)
class Colors:
  WHITE = rl.WHITE
  DISENGAGED = rl.Color(145, 155, 149, 255)
  OVERRIDE = rl.Color(145, 155, 149, 255)
  ENGAGED = rl.Color(128, 216, 166, 255)
  DISENGAGED_BG = rl.Color(0, 0, 0, 153)
  OVERRIDE_BG = rl.Color(145, 155, 149, 204)
  ENGAGED_BG = rl.Color(128, 216, 166, 204)
  GREY = rl.Color(166, 166, 166, 255)
  DARK_GREY = rl.Color(114, 114, 114, 255)
  BLACK_TRANSLUCENT = rl.Color(0, 0, 0, 166)
  WHITE_TRANSLUCENT = rl.Color(255, 255, 255, 200)
  BORDER_TRANSLUCENT = rl.Color(255, 255, 255, 75)
  HEADER_GRADIENT_START = rl.Color(0, 0, 0, 114)
  HEADER_GRADIENT_END = rl.BLANK


UI_CONFIG = UIConfig()
FONT_SIZES = FontSizes()
COLORS = Colors()


class HudRenderer(Widget):
  def __init__(self):
    super().__init__()
    """Initialize the HUD renderer."""
    self.is_cruise_set: bool = False
    self.is_cruise_available: bool = True
    self.set_speed: float = SET_SPEED_NA
    self.speed: float = 0.0
    self.v_ego_cluster_seen: bool = False
    
    # 前車距離字串與數值變數
    self.lead_dist: str = "-"
    self.lead_dist_raw: float = 0.0

    # dp: 右下角資訊列 - 方向盤角度（取代原本的駕駛/自動控制狀態文字）
    self.steering_angle_deg: float = 0.0

    # --- 紅綠燈減速中 ---
    self.ts_braking: bool = False
    self._ts_last_braking_t: float = -1e9

    self._font_semi_bold: rl.Font = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self._font_medium: rl.Font = gui_app.font(FontWeight.MEDIUM)

    self._exp_button: ExpButton = ExpButton(UI_CONFIG.button_size, UI_CONFIG.wheel_icon_size)

    # dp - jetlink
    self._jetlink_icons = jetlink_ui.JetlinkIcons(JETLINK_BADGE_W, JETLINK_BADGE_H)

    #self._torque_bar = TorqueBar(scale=4.0)

    # Lincoln perf overlay init
    self._perf_font = gui_app.font(FontWeight.MEDIUM)
    self._perf_stats: dict[str, str] = {"cpu_temp": "-", "mem_usage": "-", "disk_free": "-"}
    self._perf_lock = threading.Lock()
    self._perf_running = True
    self._perf_thread = threading.Thread(target=self._perf_update_loop, daemon=True)
    self._perf_thread.start()

  def _update_state(self) -> None:
    """Update HUD state based on car state and controls state."""
    sm = ui_state.sm
    if sm.recv_frame["carState"] < ui_state.started_frame:
      self.is_cruise_set = False
      self.set_speed = SET_SPEED_NA
      self.speed = 0.0
      self.lead_dist = "-"
      self.lead_dist_raw = 0.0
      self.steering_angle_deg = 0.0

      self.ts_braking = False
      self._ts_last_braking_t = -1e9
      return

    self._update_traffic_stop(sm)

    controls_state = sm['controlsState']
    car_state = sm['carState']
    radar_state = sm['radarState']

    # 紀錄真實距離並格式化字串
    if radar_state.leadOne.status:
      self.lead_dist_raw = radar_state.leadOne.dRel
      self.lead_dist = f"{self.lead_dist_raw:.0f}m"
    else:
      self.lead_dist_raw = 0.0
      self.lead_dist = "-"

    v_cruise_cluster = car_state.vCruiseCluster
    self.set_speed = (
      controls_state.deprecated.vCruise if v_cruise_cluster == 0.0 else v_cruise_cluster
    )
    self.is_cruise_set = 0 < self.set_speed < SET_SPEED_NA
    self.is_cruise_available = self.set_speed != -1

    if self.is_cruise_set and not ui_state.is_metric:
      self.set_speed *= KM_TO_MILE

    v_ego_cluster = car_state.vEgoCluster
    self.v_ego_cluster_seen = self.v_ego_cluster_seen or v_ego_cluster != 0.0
    v_ego = v_ego_cluster if self.v_ego_cluster_seen else car_state.vEgo
    speed_conversion = CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH
    self.speed = max(0.0, v_ego * speed_conversion)

    # dp: 右下角資訊列 - 方向盤角度
    self.steering_angle_deg = car_state.steeringAngleDeg

  def _update_traffic_stop(self, sm) -> None:
    """紅綠燈減速中：讀 longitudinalPlanDP.targets[trafficStop].braking，且 openpilot 正在控制縱向"""
    braking = False
    if sm.recv_frame['longitudinalPlanDP'] >= ui_state.started_frame and sm.alive['longitudinalPlanDP']:
      targets = sm['longitudinalPlanDP'].targets
      if len(targets) > TRAFFIC_STOP_IDX:
        braking = bool(targets[TRAFFIC_STOP_IDX].braking) and sm['carControl'].longActive

    now = rl.get_time()
    if braking:
      self._ts_last_braking_t = now
    self.ts_braking = (now - self._ts_last_braking_t) < TS_HOLD_TIME

  def _render(self, rect: rl.Rectangle) -> None:
    """Render HUD elements to the screen."""
    # Draw the header background
    rl.draw_rectangle_gradient_v(
      int(rect.x),
      int(rect.y),
      int(rect.width),
      UI_CONFIG.header_height,
      COLORS.HEADER_GRADIENT_START,
      COLORS.HEADER_GRADIENT_END,
    )

    if self.is_cruise_available:
      self._draw_set_speed(rect)

    self._draw_current_speed(rect)

    if self.ts_braking:
      self._draw_traffic_stop(rect)

    button_x = rect.x + rect.width - UI_CONFIG.border_size - UI_CONFIG.button_size
    button_y = rect.y + UI_CONFIG.border_size
    self._exp_button.render(rl.Rectangle(button_x, button_y, UI_CONFIG.button_size, UI_CONFIG.button_size))

    # dp - jetlink: the link's state, under the button
    self._draw_jetlink_badge(button_x + UI_CONFIG.button_size / 2, button_y + UI_CONFIG.button_size + 24)
    if ui_state.jetlink_state == JetlinkState.DISCONNECTED:
      self._draw_egpu_badge(button_x + UI_CONFIG.button_size / 2, button_y + UI_CONFIG.button_size + 24)

    #if ui_state.sm['controlsState'].lateralControlState.which() != 'angleState':
      #self._torque_bar.render(rect)

    # Draw performance info at bottom
    self._draw_performance_info(rect)

  def _draw_egpu_badge(self, center_x: float, top_y: float) -> None:
    """dp - egpu: sunnypilot's onroad chestnut icon, in jetlink's badge slot (jetlink is off beside an
    eGPU). Pulses while loading, green while the big model drives (label: the model's short name),
    orange when it is not used or failed this drive. Nothing without an eGPU."""
    state = ui_state.egpu_state
    icon, opacity = egpu_ui.icon_for(self._jetlink_icons, state)
    if icon is None:
      return
    bg = rl.Rectangle(center_x - JETLINK_BADGE_W / 2 - 14, top_y - 10, JETLINK_BADGE_W + 28, JETLINK_BADGE_H + JETLINK_LABEL_SIZE + 30)
    rl.draw_rectangle_rounded(bg, 0.25, 10, COLORS.BLACK_TRANSLUCENT)
    rl.draw_texture_ex(icon, rl.Vector2(center_x - icon.width / 2, top_y), 0.0, 1.0, jetlink_ui.tint(opacity))
    label = egpu_ui.badge_text(state)
    color = egpu_ui.STATE_COLOR[state]
    if state == EgpuState.LOADING:
      color = rl.Color(color.r, color.g, color.b, int(255 * max(0.5, opacity)))
    font_size = JETLINK_LABEL_SIZE
    size = measure_text_cached(self._font_semi_bold, label, font_size)
    if size.x > bg.width - 8:  # a long model short name (e.g. TTTTFBRL)
      font_size = int(JETLINK_LABEL_SIZE * 0.75)
      size = measure_text_cached(self._font_semi_bold, label, font_size)
    rl.draw_text_ex(self._font_semi_bold, label, rl.Vector2(center_x - size.x / 2, top_y + JETLINK_BADGE_H + 8),
                    font_size, 0, color)

  def _draw_jetlink_badge(self, center_x: float, top_y: float) -> None:
    """dp - jetlink: the icon (pulsing while it joins, dim green while it waits for a
    window to switch, solid green while the big model drives, orange when it cannot)
    and one word under it. Nothing without a link."""
    state = ui_state.jetlink_state
    icon, opacity = self._jetlink_icons.for_state(state)
    if icon is None:
      return
    bg = rl.Rectangle(center_x - JETLINK_BADGE_W / 2 - 14, top_y - 10, JETLINK_BADGE_W + 28, JETLINK_BADGE_H + JETLINK_LABEL_SIZE + 30)
    rl.draw_rectangle_rounded(bg, 0.25, 10, COLORS.BLACK_TRANSLUCENT)
    rl.draw_texture_ex(icon, rl.Vector2(center_x - icon.width / 2, top_y), 0.0, 1.0, jetlink_ui.tint(opacity))
    label = jetlink_ui.STATE_TEXT[state]
    color = jetlink_ui.STATE_COLOR[state]
    if state == JetlinkState.LOADING:
      color = rl.Color(color.r, color.g, color.b, int(255 * max(0.5, opacity)))
    size = measure_text_cached(self._font_semi_bold, label, JETLINK_LABEL_SIZE)
    rl.draw_text_ex(self._font_semi_bold, label, rl.Vector2(center_x - size.x / 2, top_y + JETLINK_BADGE_H + 8),
                    JETLINK_LABEL_SIZE, 0, color)

  def user_interacting(self) -> bool:
    return self._exp_button.is_pressed

  def _draw_traffic_stop(self, rect: rl.Rectangle) -> None:
    """中央車速正下方：大型紅綠燈圖示 + 「紅綠燈減速中」"""
    text = "紅綠燈減速中"
    text_size = measure_text_cached(self._font_bold, text, TS_FONT_SIZE)
    icon_w = TS_LIGHT_RADIUS * 6 + TS_LIGHT_GAP * 4
    icon_h = TS_LIGHT_RADIUS * 2 + TS_LIGHT_GAP * 2
    banner_w = TS_PADDING_X * 2 + icon_w + TS_ICON_TEXT_GAP + text_size.x

    x = rect.x + rect.width / 2 - banner_w / 2
    y = rect.y + UI_CONFIG.header_height + TS_TOP_OFFSET

    banner = rl.Rectangle(x, y, banner_w, TS_BANNER_HEIGHT)
    rl.draw_rectangle_rounded(banner, 0.3, 10, rl.Color(0, 0, 0, 190))
    border_alpha = 170 + int(85 * math.sin(rl.get_time() * 6))
    rl.draw_rectangle_rounded_lines_ex(banner, 0.3, 10, TS_BORDER, rl.Color(255, 70, 70, border_alpha))

    # 橫式紅綠燈：外框 + 紅（呼吸閃爍）/ 黃 / 綠（暗）
    icon_x = x + TS_PADDING_X
    icon_y = y + (TS_BANNER_HEIGHT - icon_h) / 2
    rl.draw_rectangle_rounded(rl.Rectangle(icon_x, icon_y, icon_w, icon_h), 0.5, 8, rl.Color(30, 30, 30, 240))
    cy = int(icon_y + icon_h / 2)
    red_alpha = 180 + int(75 * math.sin(rl.get_time() * 6))
    light_colors = (rl.Color(255, 60, 60, red_alpha), rl.Color(90, 75, 20, 255), rl.Color(20, 70, 35, 255))
    for i, color in enumerate(light_colors):
      cx = int(icon_x + TS_LIGHT_GAP + TS_LIGHT_RADIUS + i * (TS_LIGHT_RADIUS * 2 + TS_LIGHT_GAP))
      rl.draw_circle(cx, cy, TS_LIGHT_RADIUS, color)

    text_x = icon_x + icon_w + TS_ICON_TEXT_GAP
    text_y = y + (TS_BANNER_HEIGHT - text_size.y) / 2
    rl.draw_text_ex(self._font_bold, text, rl.Vector2(text_x, text_y), TS_FONT_SIZE, 0, rl.WHITE)

  def _draw_set_speed(self, rect: rl.Rectangle) -> None:
    set_speed_width = UI_CONFIG.set_speed_width_metric if ui_state.is_metric else UI_CONFIG.set_speed_width_imperial
    x = rect.x + 60 + (UI_CONFIG.set_speed_width_imperial - set_speed_width) // 2
    y = rect.y + 45

    set_speed_rect = rl.Rectangle(x, y, set_speed_width, UI_CONFIG.set_speed_height)
    rl.draw_rectangle_rounded(set_speed_rect, 0.35, 10, COLORS.BLACK_TRANSLUCENT)
    rl.draw_rectangle_rounded_lines_ex(set_speed_rect, 0.35, 10, 6, COLORS.BORDER_TRANSLUCENT)

    max_color = COLORS.GREY
    set_speed_color = COLORS.DARK_GREY
    if self.is_cruise_set:
      set_speed_color = COLORS.WHITE
      if ui_state.status == UIStatus.ENGAGED:
        max_color = COLORS.ENGAGED
      elif ui_state.status == UIStatus.DISENGAGED:
        max_color = COLORS.DISENGAGED
      elif ui_state.status == UIStatus.OVERRIDE:
        max_color = COLORS.OVERRIDE

    max_text = tr("MAX")
    max_text_width = measure_text_cached(self._font_semi_bold, max_text, FONT_SIZES.max_speed).x
    rl.draw_text_ex(
      self._font_semi_bold,
      max_text,
      rl.Vector2(x + (set_speed_width - max_text_width) / 2, y + 27),
      FONT_SIZES.max_speed,
      0,
      max_color,
    )

    set_speed_text = CRUISE_DISABLED_CHAR if not self.is_cruise_set else str(round(self.set_speed))
    speed_text_width = measure_text_cached(self._font_bold, set_speed_text, FONT_SIZES.set_speed).x
    rl.draw_text_ex(
      self._font_bold,
      set_speed_text,
      rl.Vector2(x + (set_speed_width - speed_text_width) / 2, y + 77),
      FONT_SIZES.set_speed,
      0,
      set_speed_color,
    )

  def _draw_current_speed(self, rect: rl.Rectangle) -> None:
    speed_text = str(round(self.speed))
    speed_text_size = measure_text_cached(self._font_bold, speed_text, FONT_SIZES.current_speed)
    speed_pos = rl.Vector2(rect.x + rect.width / 2 - speed_text_size.x / 2, 180 - speed_text_size.y / 2)
    rl.draw_text_ex(self._font_bold, speed_text, speed_pos, FONT_SIZES.current_speed, 0, COLORS.WHITE)

    unit_text = tr("km/h") if ui_state.is_metric else tr("mph")
    unit_text_size = measure_text_cached(self._font_medium, unit_text, FONT_SIZES.speed_unit)
    unit_pos = rl.Vector2(rect.x + rect.width / 2 - unit_text_size.x / 2, 290 - unit_text_size.y / 2)
    rl.draw_text_ex(self._font_medium, unit_text, unit_pos, FONT_SIZES.speed_unit, 0, COLORS.WHITE_TRANSLUCENT)

  def _draw_performance_info(self, rect: rl.Rectangle) -> None:
    if rect.width <= 0 or rect.height <= 0:
      return

    stats = self._get_perf_stats()
    steering_angle_text = f"{tr('Steering Angle')} {self.steering_angle_deg:.0f}°"

    lead_dist = self.lead_dist
    cpu_temp = stats.get("cpu_temp", "-")
    mem_usage = stats.get("mem_usage", "-")
    disk_free = stats.get("disk_free", "-")

    items = [
      f"{tr('Lead Dist')} {lead_dist}",
      f"{tr('CPU Temp')} {cpu_temp}",
      f"{tr('Memory')} {mem_usage}",
      f"{tr('Disk Free')} {disk_free}",
      f"{steering_angle_text}",
    ]

    measurements = [measure_text_cached(self._perf_font, text, PERF_FONT_SIZE) for text in items]

    bar_width = max(rect.width - 20, 0)
    bar_height = PERF_FONT_SIZE + 2 * PERF_PADDING

    bar_x = rect.x + (rect.width - bar_width) / 2
    bar_y = rect.y + rect.height - bar_height - PERF_MARGIN_BOTTOM
    minimum_y = rect.y + PERF_MARGIN_BOTTOM
    if bar_y < minimum_y:
      bar_y = minimum_y

    rl.draw_rectangle_rounded(
      rl.Rectangle(bar_x, bar_y, bar_width, bar_height),
      0.2,
      8,
      PERF_BG_COLOR,
    )

    slot_width = bar_width / len(items)
    text_y = bar_y + PERF_PADDING

    for i, (text, measurement) in enumerate(zip(items, measurements)):
      cursor_x = bar_x + (i * slot_width) + (slot_width - measurement.x) / 2
      text_color = rl.WHITE
      if i == 0 and self.lead_dist != "-" and self.lead_dist_raw < 15.0:
        text_color = rl.Color(255, 188, 0, 200)

      rl.draw_text_ex(self._perf_font, text, rl.Vector2(cursor_x, text_y), PERF_FONT_SIZE, 0, text_color)

  def _get_perf_stats(self) -> dict[str, str]:
    with self._perf_lock:
      return dict(self._perf_stats)

  def _perf_update_loop(self) -> None:
    time.sleep(10)
    while self._perf_running:
      stats = {
        "cpu_temp": self._read_cpu_temp(),
        "mem_usage": self._read_mem_usage(),
        "disk_free": self._read_disk_free(),
      }
      with self._perf_lock:
        self._perf_stats.update(stats)
      for _ in range(10):
        if not self._perf_running:
          return
        time.sleep(0.1)

  @staticmethod
  def _read_cpu_temp() -> str:
    path = "/sys/class/thermal/thermal_zone0/temp"
    try:
      with open(path) as f:
        temp_c = int(f.read().strip()) / 1000.0
        return f"{temp_c:.0f}°C"
    except Exception:
      return "-"

  @staticmethod
  def _read_mem_usage() -> str:
    try:
      total_kb = None
      available_kb = None
      with open("/proc/meminfo") as f:
        for line in f:
          if line.startswith("MemTotal:"):
            total_kb = float(line.split()[1])
          elif line.startswith("MemAvailable:"):
            available_kb = float(line.split()[1])
          if total_kb is not None and available_kb is not None:
            break
      if total_kb and available_kb:
        used_pct = (total_kb - available_kb) / total_kb * 100.0
        used_pct = min(max(used_pct, 0.0), 100.0)
        return f"{used_pct:.0f}%"
    except Exception:
      pass
    return "-"

  @staticmethod
  def _read_disk_free() -> str:
    try:
      usage = shutil.disk_usage("/data")
      free_gb = usage.free / (1024 ** 3)
      if free_gb >= 1.0:
        return f"{free_gb:.1f}G" 
      free_mb = usage.free / (1024 ** 2)
      return f"{free_mb:.0f}MB"  
    except Exception:
      return "-"
