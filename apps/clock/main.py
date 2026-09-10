#!/usr/bin/env python3

import json
import mmap
import os
import select
import struct
import subprocess
import threading
import time
import queue
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------
# Display constants (from Sipeed's NanoKVM-Desk DIY app documentation)
# --------------------------------------------------------------------------
PHYSICAL_WIDTH = 172
PHYSICAL_HEIGHT = 320
BPP = 16  # RGB565

LOGICAL_WIDTH = 320
LOGICAL_HEIGHT = 172

# --------------------------------------------------------------------------
# Input device constants
# --------------------------------------------------------------------------
TOUCH_DEVICE = "/dev/input/event2"
ROTARY_DEVICE = "/dev/input/event0"
KNOB_BUTTON_DEVICE = "/dev/input/event1"

EVENT_FORMAT = "qqHHi"          # see hardware note above
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)

EV_KEY = 1
EV_REL = 2
EV_ABS = 3

BTN_TOUCH = 330
REL_X = 0
KEY_ENTER = 28

LONG_PRESS_EXIT_SEC = 3.0
MAIN_LOOP_TICK_SEC = 0.1

# --------------------------------------------------------------------------
# Remote-session watchdog configuration
# --------------------------------------------------------------------------
SESSION_EXCLUDE_PORTS = {22}
SESSION_CHECK_INTERVAL_SEC = 3
LOOPBACK_ADDRESSES = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}

# --------------------------------------------------------------------------
# Color themes and brightness levels
# Each theme: (name, digit_color, colon_color, off_segment_color)
# Colors are kept muted/desaturated by design -- the goal is a pleasant,
# low-glare panel, not a bright showpiece.
# --------------------------------------------------------------------------
BG_COLOR = (6, 6, 8)  # near-black background, not themed/scaled

THEMES = [
    ("Amber", (168, 104, 40), (120, 74, 30), (16, 14, 12)),
    ("Red",   (170, 50, 50),  (120, 35, 35), (14, 10, 10)),
    ("Teal",  (40, 140, 140), (30, 100, 100), (10, 14, 14)),
    ("Green", (60, 150, 70),  (40, 110, 50), (10, 14, 10)),
    ("Blue",  (70, 110, 180), (45, 75, 125), (10, 12, 16)),
]

# (label, multiplier). 1.0x reproduces the original reference colors above.
BRIGHTNESS_LEVELS = [
    ("Dim", 0.6),
    ("Normal", 1.0),
    ("Bright", 1.5),
]

EXIT_BAR_COLOR = (150, 40, 40)       # fixed regardless of theme -- a "warning" cue
REMOTE_ACTIVE_COLOR = (200, 30, 30)  # fixed red regardless of theme, per spec
REMOTE_ACTIVE_SCALE = 3              # largest size that clears the digits (see module docstring math)

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_SETTINGS = {
    "time_format": "24h",       # "24h" or "12h"
    "brightness_index": 1,      # index into BRIGHTNESS_LEVELS
    "colon_blink": True,
    "theme_index": 0,           # index into THEMES
}


def load_settings():
    try:
        with open(SETTINGS_PATH, "r") as f:
            data = json.load(f)
    except Exception:
        return DEFAULT_SETTINGS.copy()

    settings = DEFAULT_SETTINGS.copy()
    if data.get("time_format") in ("24h", "12h"):
        settings["time_format"] = data["time_format"]
    if isinstance(data.get("brightness_index"), int) and 0 <= data["brightness_index"] < len(BRIGHTNESS_LEVELS):
        settings["brightness_index"] = data["brightness_index"]
    if isinstance(data.get("colon_blink"), bool):
        settings["colon_blink"] = data["colon_blink"]
    if isinstance(data.get("theme_index"), int) and 0 <= data["theme_index"] < len(THEMES):
        settings["theme_index"] = data["theme_index"]
    return settings


def save_settings(settings):
    try:
        tmp_path = SETTINGS_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(settings, f)
        os.replace(tmp_path, SETTINGS_PATH)
    except Exception:
        pass  # persistence is a nice-to-have; don't crash the app over it


def scale_color(color, factor):
    return tuple(min(255, max(0, int(round(c * factor)))) for c in color)


def compute_colors(settings):
    _name, digit, colon, off = THEMES[settings["theme_index"]]
    factor = BRIGHTNESS_LEVELS[settings["brightness_index"]][1]
    return {
        "digit": scale_color(digit, factor),
        "colon": scale_color(colon, factor),
        "off": scale_color(off, factor),
    }


# --------------------------------------------------------------------------
# Seven-segment digit definitions
# Segments: a(top), b(upper-right), c(lower-right), d(bottom),
#           e(lower-left), f(upper-left), g(middle)
# --------------------------------------------------------------------------
DIGIT_SEGMENTS = {
    "0": set("abcdef"),
    "1": set("bc"),
    "2": set("abged"),
    "3": set("abgcd"),
    "4": set("fgbc"),
    "5": set("afgcd"),
    "6": set("afgecd"),
    "7": set("abc"),
    "8": set("abcdefg"),
    "9": set("abcdfg"),
}


def segment_boxes(x, y, w, h):
    """Return a dict of segment-letter -> (x0, y0, x1, y1) for a digit box."""
    t = max(4, int(w * 0.20))          # segment thickness
    half_gap = t * 0.5
    return {
        "a": (x + t, y, x + w - t, y + t),
        "f": (x, y + t, x + t, y + h / 2 - half_gap),
        "b": (x + w - t, y + t, x + w, y + h / 2 - half_gap),
        "g": (x + t, y + h / 2 - t / 2, x + w - t, y + h / 2 + t / 2),
        "e": (x, y + h / 2 + half_gap, x + t, y + h - t),
        "c": (x + w - t, y + h / 2 + half_gap, x + w, y + h - t),
        "d": (x + t, y + h - t, x + w - t, y + h),
    }


class RGB565Display:
    """Thin wrapper around /dev/fb0 for the NanoKVM-Desk screen."""

    def __init__(self, fb_device="/dev/fb0"):
        self.physical_width = PHYSICAL_WIDTH
        self.physical_height = PHYSICAL_HEIGHT
        self.bpp = BPP
        self.fb_size = self.physical_width * self.physical_height * (self.bpp // 8)

        self.fb_fd = os.open(fb_device, os.O_RDWR)
        self.fb_mmap = mmap.mmap(
            self.fb_fd, self.fb_size, mmap.MAP_SHARED, mmap.PROT_WRITE
        )
        self.fb_array = np.frombuffer(self.fb_mmap, dtype=np.uint16).reshape(
            (self.physical_height, self.physical_width)
        )

    def display_image(self, logical_img):
        """Rotate the logical (320x172) image and blit it to the physical screen."""
        physical_img = logical_img.rotate(90, expand=True)

        rgb_array = np.array(physical_img)
        r = (rgb_array[:, :, 0] >> 3).astype(np.uint16)
        g = (rgb_array[:, :, 1] >> 2).astype(np.uint16)
        b = (rgb_array[:, :, 2] >> 3).astype(np.uint16)
        rgb565 = (r << 11) | (g << 5) | b

        self.fb_array[:, :] = rgb565

    def close(self):
        try:
            self.fb_mmap.close()
        finally:
            os.close(self.fb_fd)


# --------------------------------------------------------------------------
# Remote-session watchdog
# --------------------------------------------------------------------------
class SessionWatchdog:

    def __init__(self, exclude_ports, interval_sec):
        self.exclude_ports = exclude_ports
        self.interval_sec = interval_sec

        self._lock = threading.Lock()
        self._is_active = False
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    @property
    def is_active(self):
        with self._lock:
            return self._is_active

    @staticmethod
    def _split_addr_port(endpoint):
        """Split an 'addr:port' (or '[ipv6]:port') string from ss output."""
        endpoint = endpoint.strip()
        addr, _, port = endpoint.rpartition(":")
        return addr.strip("[]"), port

    @staticmethod
    def _run_ss(*args):
        try:
            result = subprocess.run(
                ["ss", "-H", *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            return result.stdout.splitlines()
        except Exception:
            return []

    def _get_listening_ports(self):
        """Return the set of local TCP ports currently in LISTEN state,
        minus exclude_ports."""
        ports = set()
        for line in self._run_ss("-tln"):
            fields = line.split()
            if len(fields) < 4:
                continue
            _addr, port = self._split_addr_port(fields[-2])
            if port.isdigit() and int(port) not in self.exclude_ports:
                ports.add(int(port))
        return ports

    def _has_remote_session(self):
        listening_ports = self._get_listening_ports()
        if not listening_ports:
            return False

        for line in self._run_ss("-tn", "state", "established"):
            fields = line.split()
            if len(fields) < 4:
                continue
            _local_addr, local_port = self._split_addr_port(fields[-2])
            peer_addr, _peer_port = self._split_addr_port(fields[-1])

            if not local_port.isdigit() or int(local_port) not in listening_ports:
                continue
            if peer_addr in LOOPBACK_ADDRESSES:
                continue

            return True

        return False

    def _run(self):
        while not self._stop_event.is_set():
            active = self._has_remote_session()
            with self._lock:
                self._is_active = active
            self._stop_event.wait(self.interval_sec)


# --------------------------------------------------------------------------
# Touch input (used only for the long-press-to-exit gesture)
# --------------------------------------------------------------------------
class TouchInput:

    def __init__(self, device=TOUCH_DEVICE, long_press_sec=LONG_PRESS_EXIT_SEC):
        self.device = device
        self.long_press_sec = long_press_sec

        self._lock = threading.Lock()
        self._is_down = False
        self._down_since = None

        self._exit_requested = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    @property
    def exit_requested(self):
        return self._exit_requested.is_set()

    def get_hold_progress(self):
        with self._lock:
            if self._is_down and self._down_since is not None:
                elapsed = time.monotonic() - self._down_since
                if elapsed >= self.long_press_sec:
                    self._exit_requested.set()
                return min(1.0, elapsed / self.long_press_sec)
        return 0.0

    def _run(self):
        try:
            fd = os.open(self.device, os.O_RDONLY)
        except Exception:
            # No touchscreen available -- exit gesture just won't work;
            # the clock still runs fine.
            return

        try:
            while not self._stop_event.is_set():
                ready, _, _ = select.select([fd], [], [], 0.5)
                if not ready:
                    continue
                data = os.read(fd, EVENT_SIZE)
                if len(data) < EVENT_SIZE:
                    continue
                _sec, _usec, ev_type, ev_code, ev_value = struct.unpack(EVENT_FORMAT, data)

                if ev_type == EV_KEY and ev_code == BTN_TOUCH:
                    if ev_value == 1:
                        with self._lock:
                            self._is_down = True
                            self._down_since = time.monotonic()
                    elif ev_value == 0:
                        with self._lock:
                            self._is_down = False
                            self._down_since = None
        finally:
            os.close(fd)


# --------------------------------------------------------------------------
# Rotary knob input (rotation + press), used to drive the Settings screen
# --------------------------------------------------------------------------
class KnobInput:

    ROTATION_UNITS_PER_STEP = 2

    def __init__(self, rotation_device=ROTARY_DEVICE, button_device=KNOB_BUTTON_DEVICE):
        self.rotation_device = rotation_device
        self.button_device = button_device

        self._lock = threading.Lock()
        self._rotation_accum = 0  # raw pulse units not yet converted to a full step
        self._press_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def pop_rotation_delta(self):
        with self._lock:
            accum = self._rotation_accum
            steps = int(accum / self.ROTATION_UNITS_PER_STEP)  # truncates toward zero
            self._rotation_accum = accum - steps * self.ROTATION_UNITS_PER_STEP
        return steps

    def pop_press(self):
        try:
            self._press_queue.get_nowait()
            return True
        except queue.Empty:
            return False

    def _accumulate_rotation(self, ev_value):
        with self._lock:
            if self._rotation_accum != 0 and (ev_value > 0) != (self._rotation_accum > 0):
                self._rotation_accum = 0
            self._rotation_accum += ev_value

    def _run(self):
        fds = {}
        try:
            fds[os.open(self.rotation_device, os.O_RDONLY)] = "rotation"
        except Exception:
            pass
        try:
            fds[os.open(self.button_device, os.O_RDONLY)] = "button"
        except Exception:
            pass

        if not fds:
            # No knob hardware available -- Settings becomes unreachable,
            # but the clock itself still runs fine.
            return

        try:
            while not self._stop_event.is_set():
                ready, _, _ = select.select(list(fds.keys()), [], [], 0.5)
                for fd in ready:
                    data = os.read(fd, EVENT_SIZE)
                    if len(data) < EVENT_SIZE:
                        continue
                    _sec, _usec, ev_type, ev_code, ev_value = struct.unpack(EVENT_FORMAT, data)

                    if fds[fd] == "rotation" and ev_type == EV_REL and ev_code == REL_X:
                        self._accumulate_rotation(ev_value)
                    elif fds[fd] == "button" and ev_type == EV_KEY and ev_code == KEY_ENTER:
                        if ev_value == 1:  # initial press only; ignore repeat (2) and release (0)
                            self._press_queue.put(True)
        finally:
            for fd in fds:
                try:
                    os.close(fd)
                except Exception:
                    pass


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def render_pixel_text(text, color, scale=2):
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = probe.textbbox((0, 0), text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 1

    small = Image.new("RGB", (text_w + pad * 2, text_h + pad * 2), (0, 0, 0))
    d = ImageDraw.Draw(small)
    d.text((pad - bbox[0], pad - bbox[1]), text, fill=color)

    return small.resize((small.width * scale, small.height * scale), Image.NEAREST)


def paste_text(img, text, color, xy, scale=2):
    """Render text and paste it (transparent-black background) at xy."""
    label = render_pixel_text(text, color, scale)
    img.paste(label, xy)
    return label.size


def draw_digit(draw, x, y, w, h, digit_char, colors):
    boxes = segment_boxes(x, y, w, h)
    lit = DIGIT_SEGMENTS[digit_char]
    for seg, box in boxes.items():
        color = colors["digit"] if seg in lit else colors["off"]
        draw.rectangle(box, fill=color)


def draw_colon(draw, x, y, h, visible, colors):
    """Draw the separator between HH and MM."""
    dot = max(5, int(h * 0.09))
    cx = x
    cy_top = y + h * 0.30
    cy_bot = y + h * 0.70
    color = colors["colon"] if visible else colors["off"]
    draw.ellipse([cx - dot / 2, cy_top - dot / 2, cx + dot / 2, cy_top + dot / 2], fill=color)
    draw.ellipse([cx - dot / 2, cy_bot - dot / 2, cx + dot / 2, cy_bot + dot / 2], fill=color)


# Digit block geometry, computed once and reused by both the clock and the
# REMOTE ACTIVE placement math so the two can never accidentally overlap.
DIGIT_W, DIGIT_H = 60, 96
DIGIT_Y = (LOGICAL_HEIGHT - DIGIT_H) // 2
DIGIT_BOTTOM = DIGIT_Y + DIGIT_H


def draw_clock_frame(now, settings):
    """Build one logical (320x172) frame showing the current time."""
    colors = compute_colors(settings)
    img = Image.new("RGB", (LOGICAL_WIDTH, LOGICAL_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    fmt = "%I%M" if settings["time_format"] == "12h" else "%H%M"
    digits = now.strftime(fmt)

    if settings["colon_blink"]:
        colon_visible = (now.second % 2 == 0)
    else:
        colon_visible = True

    colon_w = 30
    gap = 8

    total_w = DIGIT_W * 4 + colon_w + gap * 4
    start_x = (LOGICAL_WIDTH - total_w) // 2

    x = start_x
    draw_digit(draw, x, DIGIT_Y, DIGIT_W, DIGIT_H, digits[0], colors)
    x += DIGIT_W + gap
    draw_digit(draw, x, DIGIT_Y, DIGIT_W, DIGIT_H, digits[1], colors)
    x += DIGIT_W + gap

    draw_colon(draw, x + colon_w / 2, DIGIT_Y, DIGIT_H, colon_visible, colors)
    x += colon_w + gap

    draw_digit(draw, x, DIGIT_Y, DIGIT_W, DIGIT_H, digits[2], colors)
    x += DIGIT_W + gap
    draw_digit(draw, x, DIGIT_Y, DIGIT_W, DIGIT_H, digits[3], colors)

    if settings["time_format"] == "12h":
        ampm = now.strftime("%p")
        margin = 6
        label_img = render_pixel_text(ampm, colors["off"], scale=2)
        img.paste(label_img, (LOGICAL_WIDTH - label_img.width - margin, margin))

    return img


def draw_remote_active_overlay(img):
    """Overlay 'REMOTE ACTIVE' in red under the clock, sized as large as
    possible while staying clear of the digits (see DIGIT_BOTTOM)."""
    label_img = render_pixel_text("REMOTE ACTIVE", REMOTE_ACTIVE_COLOR, scale=REMOTE_ACTIVE_SCALE)
    available = LOGICAL_HEIGHT - DIGIT_BOTTOM
    y = DIGIT_BOTTOM + max(0, (available - label_img.height) // 2)
    x = (LOGICAL_WIDTH - label_img.width) // 2
    img.paste(label_img, (x, y))


# --------------------------------------------------------------------------
# Settings screen (knob-driven)
# --------------------------------------------------------------------------
SETTINGS_ROWS = ["format", "brightness", "blink", "theme", "back"]

ROW_X0, ROW_X1 = 10, 310
ROW_H = 25   # fits the tallest label (descenders in words like "Brightness") with a hair of padding
ROW_GAP = 1
ROW_START_Y = 24  # leaves room for the title above


def _row_label(row_id, settings):
    if row_id == "format":
        value = "24-hour" if settings["time_format"] == "24h" else "12-hour"
        return "Format: " + value
    if row_id == "brightness":
        return "Brightness: " + BRIGHTNESS_LEVELS[settings["brightness_index"]][0]
    if row_id == "blink":
        return "Colon Blink: " + ("On" if settings["colon_blink"] else "Off")
    if row_id == "theme":
        return "Theme: " + THEMES[settings["theme_index"]][0]
    if row_id == "back":
        return "Back"
    return row_id


def _row_display_label(row_id, settings, editing):
    label = _row_label(row_id, settings)
    if editing and ": " in label:
        head, _, value = label.partition(": ")
        return "{}: <{}>".format(head, value)
    return label


def get_row_value_count(row_id):
    if row_id == "format":
        return 2
    if row_id == "brightness":
        return len(BRIGHTNESS_LEVELS)
    if row_id == "blink":
        return 2
    if row_id == "theme":
        return len(THEMES)
    return 1  # "back" has no cyclable value


def get_row_index(row_id, settings):
    if row_id == "format":
        return 0 if settings["time_format"] == "24h" else 1
    if row_id == "brightness":
        return settings["brightness_index"]
    if row_id == "blink":
        return 1 if settings["colon_blink"] else 0
    if row_id == "theme":
        return settings["theme_index"]
    return 0


def set_row_index(row_id, settings, idx):
    if row_id == "format":
        settings["time_format"] = "24h" if idx == 0 else "12h"
    elif row_id == "brightness":
        settings["brightness_index"] = idx
    elif row_id == "blink":
        settings["colon_blink"] = bool(idx)
    elif row_id == "theme":
        settings["theme_index"] = idx
    # "back": no-op, nothing to cycle


def draw_settings_frame(settings, selected_row, mode):
    colors = compute_colors(settings)
    img = Image.new("RGB", (LOGICAL_WIDTH, LOGICAL_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    title_img = render_pixel_text("SETTINGS", colors["digit"], scale=2)
    img.paste(title_img, ((LOGICAL_WIDTH - title_img.width) // 2, 2))

    y = ROW_START_Y
    for i, row_id in enumerate(SETTINGS_ROWS):
        rect = (ROW_X0, y, ROW_X1, y + ROW_H)
        is_selected = (i == selected_row)
        editing = is_selected and mode == "edit"

        if is_selected:
            draw.rectangle(rect, fill=colors["off"])
            draw.rectangle(rect, outline=colors["digit"], width=2 if editing else 1)
            label = "> " + _row_display_label(row_id, settings, editing)
        else:
            draw.rectangle(rect, outline=colors["off"], width=1)
            label = _row_display_label(row_id, settings, False)

        label_img = render_pixel_text(label, colors["digit"], scale=2)
        text_x = rect[0] + 6
        text_y = rect[1] + max(0, (ROW_H - label_img.height) // 2)
        img.paste(label_img, (text_x, text_y))

        y += ROW_H + ROW_GAP

    if mode == "edit":
        hint = "Rotate: change value   Press: confirm"
    else:
        hint = "Rotate: select row   Press: choose"
    hint_img = render_pixel_text(hint, colors["off"], scale=1)
    img.paste(hint_img, ((LOGICAL_WIDTH - hint_img.width) // 2, y + 5))

    return img


def draw_hold_progress_overlay(img, progress):
    """Thin bar along the bottom edge showing progress toward the
    long-press-to-exit threshold."""
    draw = ImageDraw.Draw(img)
    bar_w = int(LOGICAL_WIDTH * min(1.0, progress))
    if bar_w > 0:
        draw.rectangle([0, LOGICAL_HEIGHT - 3, bar_w, LOGICAL_HEIGHT - 1], fill=EXIT_BAR_COLOR)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    display = RGB565Display()
    watchdog = SessionWatchdog(SESSION_EXCLUDE_PORTS, SESSION_CHECK_INTERVAL_SEC)
    watchdog.start()
    touch = TouchInput()
    touch.start()
    knob = KnobInput()
    knob.start()

    settings = load_settings()
    screen = "clock"          # "clock" or "settings"
    settings_mode = "navigate"  # "navigate" or "edit"
    selected_row = 0

    try:
        while True:
            if touch.exit_requested:
                break

            if knob.pop_press():
                if screen == "clock":
                    screen = "settings"
                    settings_mode = "navigate"
                    selected_row = 0
                else:
                    row_id = SETTINGS_ROWS[selected_row]
                    if settings_mode == "navigate":
                        if row_id == "back":
                            screen = "clock"
                            save_settings(settings)
                        else:
                            settings_mode = "edit"
                    else:  # currently editing -- confirm and go back to navigating
                        settings_mode = "navigate"

            delta = knob.pop_rotation_delta()
            if delta and screen == "settings":
                row_id = SETTINGS_ROWS[selected_row]
                if settings_mode == "navigate":
                    selected_row = (selected_row + delta) % len(SETTINGS_ROWS)
                else:
                    count = get_row_value_count(row_id)
                    if count > 1:
                        idx = (get_row_index(row_id, settings) + delta) % count
                        set_row_index(row_id, settings, idx)

            hold_progress = touch.get_hold_progress()
            now = datetime.now()

            if screen == "clock":
                frame = draw_clock_frame(now, settings)
                if watchdog.is_active:
                    draw_remote_active_overlay(frame)
            else:
                frame = draw_settings_frame(settings, selected_row, settings_mode)

            if hold_progress > 0.15:
                draw_hold_progress_overlay(frame, hold_progress)

            display.display_image(frame)
            time.sleep(MAIN_LOOP_TICK_SEC)

    except KeyboardInterrupt:
        pass
    finally:
        save_settings(settings)
        watchdog.stop()
        touch.stop()
        knob.stop()
        display.close()


if __name__ == "__main__":
    main()
