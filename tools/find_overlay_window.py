import ctypes
import ctypes.wintypes

user32 = ctypes.windll.user32

windows = []

def enum_cb(hwnd, lparam):
    if user32.IsWindowVisible(hwnd):
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        windows.append((hwnd, pid.value, title, (rect.left, rect.top, rect.right, rect.bottom)))
    return True

WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

for h, pid, title, r in windows:
    if "asmr" in title.lower() or "tk" in title.lower() or "sub" in title.lower() or not title:
        # 看看是不是尺寸像 1120x116 的无标题窗口
        w = r[2] - r[0]
        h_val = r[3] - r[1]
        if w == 1120 or "asmr" in title.lower():
            print(f"FOUND TARGET WINDOW: HWND={h}, PID={pid}, Title='{title}', Rect={r}, Size={w}x{h_val}")
