import ctypes
import ctypes.wintypes

user32 = ctypes.windll.user32

monitors = []

def monitor_enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData):
    r = lprcMonitor.contents
    monitors.append((r.left, r.top, r.right, r.bottom))
    return True

MONITORENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HMONITOR, ctypes.wintypes.HDC, ctypes.POINTER(ctypes.wintypes.RECT), ctypes.wintypes.LPARAM)
user32.EnumDisplayMonitors(0, 0, MONITORENUMPROC(monitor_enum_proc), 0)

print(f"检测到 {len(monitors)} 个显示器:")
for i, (l, t, r, b) in enumerate(monitors):
    w = r - l
    h = b - t
    print(f"显示器 [{i}]: 左={l}, 上={t}, 右={r}, 下={b} (分辨率: {w}x{h})")
