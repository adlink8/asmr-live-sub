"""Overlay subtitle window (tkinter + Win32).

Borderless, semi-transparent, draggable; auto-placed on the external monitor
when one exists. Re-topmost is re-asserted on every poll tick because
overrideredirect windows lose z-order to fullscreen apps on Windows 11.

Note: ctypes is imported at module level on purpose. _force_topmost() runs
from the 120ms poll callback; a function-local import in __init__ would leave
the global name unbound there and the SetWindowPos call would die as a
NameError inside the bare except — silently degrading to tkinter -topmost.
"""
import ctypes
import queue


def choose_monitor(screen_idx=None) -> tuple[int, int, int, int]:
    """Enumerate physical monitors and pick one: (left, top, right, bottom).

    Explicit --screen index wins; otherwise the external monitor [1] is
    preferred so the overlay lands on the big screen the user watches.
    Falls back to 1920x1080 if enumeration yields nothing.
    """
    import ctypes.wintypes

    user32 = ctypes.windll.user32
    monitors: list[tuple[int, int, int, int]] = []

    def _m_enum(hM, hdc, lprc, p):
        r = lprc.contents
        monitors.append((r.left, r.top, r.right, r.bottom))
        return True

    proc_t = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HMONITOR,
                                ctypes.wintypes.HDC,
                                ctypes.POINTER(ctypes.wintypes.RECT),
                                ctypes.wintypes.LPARAM)
    user32.EnumDisplayMonitors(0, 0, proc_t(_m_enum), 0)

    if screen_idx is not None and 0 <= screen_idx < len(monitors):
        return monitors[screen_idx]
    if len(monitors) > 1:
        return monitors[1]
    if monitors:
        return monitors[0]
    return (0, 0, 1920, 1080)


class SubtitleWindow:
    def __init__(self, result_q, on_close, screen_idx=None, stream_q=None):
        import tkinter as tk
        self.q = result_q
        self.stream_q = stream_q
        self.stream_buf = []
        self.on_close = on_close
        self.root = tk.Tk()
        self.root.title("asmr-live-sub")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.90)

        # 多屏定位：默认若存在外接屏(>1)，优先放在外接大屏[1]，否则落主屏[0]
        m_left, m_top, m_right, m_bottom = choose_monitor(screen_idx)
        m_w = m_right - m_left
        m_h = m_bottom - m_top

        w, h = min(1120, int(m_w * 0.85)), 116
        self.w = w
        # 居中放置在当前屏幕底部
        x = m_left + (m_w - w) // 2
        y = m_top + m_h - h - 140
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        # 带 1px 亮蓝色高对比度边框与深黑底色
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="#0b0f19",
                                highlightthickness=1, highlightbackground="#3b82f6")
        self.canvas.pack(fill="both", expand=True)
        self.t_now = self.canvas.create_text(w // 2, 42, fill="#ffffff", font=("Microsoft YaHei", 20, "bold"), width=w - 40)
        self.t_prev = self.canvas.create_text(w // 2, 88, fill="#94a3b8", font=("Microsoft YaHei", 12), width=w - 40)
        self.t_status = self.canvas.create_text(10, 8, anchor="w", fill="#22c55e", font=("Consolas", 9, "bold"))
        self.canvas.itemconfig(self.t_status, text="listening... [按住左键可拖动 | Esc退出]")

        self.root.update_idletasks()
        self.hwnd = None
        try:
            self.hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            # WS_EX_APPWINDOW 让任务栏显示图标
            gwl_exstyle = -20
            style = ctypes.windll.user32.GetWindowLongW(self.hwnd, gwl_exstyle)
            ctypes.windll.user32.SetWindowLongW(self.hwnd, gwl_exstyle, style | 0x00040000)
            self._force_topmost()
        except Exception:
            pass

        self.canvas.bind("<ButtonPress-1>", self._press)
        self.root.bind("<Escape>", lambda e: self.close())
        self._drag_off = None

    def _force_topmost(self):
        try:
            if self.hwnd:
                # HWND_TOPMOST = -1, SWP_NOMOVE=2, SWP_NOSIZE=1, SWP_SHOWWINDOW=0x40
                ctypes.windll.user32.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)
        except Exception:
            pass

    def _press(self, ev):
        self._drag_off = (ev.x_root - self.root.winfo_x(), ev.y_root - self.root.winfo_y())
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.canvas.unbind("<B1-Motion>"))

    def _drag(self, ev):
        if self._drag_off:
            self.root.geometry(f"+{ev.x_root - self._drag_off[0]}+{ev.y_root - self._drag_off[1]}")

    def poll(self):
        # 1. 消费打字机流式 token，实现 39ms 首字即时上屏
        if self.stream_q is not None:
            updated = False
            while True:
                try:
                    tok = self.stream_q.get_nowait()
                    if tok is not None:
                        self.stream_buf.append(tok)
                        updated = True
                except queue.Empty:
                    break
            if updated and self.stream_buf:
                live_text = "".join(self.stream_buf).strip()
                if live_text:
                    self.canvas.itemconfig(self.t_now, text=live_text)
                    self.canvas.itemconfig(self.t_status, text="typing...")

        # 2. 消费最终整句结果（由 MT 校验与拒答过滤后的终稿）
        while True:
            try:
                item = self.q.get_nowait()
                if len(item) >= 3:
                    zh, ja, dt = item[0], item[1], item[2]
                else:
                    zh, ja, dt = item
                self.stream_buf = []  # 重置流式缓冲
                self.canvas.itemconfig(self.t_prev, text=ja)
                self.canvas.itemconfig(self.t_now, text=zh or ja)
                self.canvas.itemconfig(self.t_status, text=f"ok {dt:.1f}s")
            except queue.Empty:
                break

        self._force_topmost()
        # 40ms 高刷（25 FPS）实现平滑逐字打字机动画
        self.root.after(40, self.poll)

    def run(self):
        self.root.after(40, self.poll)
        self.root.mainloop()
        self.on_close.set()

    def close(self):
        self.root.destroy()
