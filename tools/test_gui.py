import tkinter as tk
import time

root = tk.Tk()
root.title("TEST_DESKTOP_WINDOW")
root.geometry("400x200+500+300")
root.attributes("-topmost", True)
lbl = tk.Label(root, text="如果你能看到这个窗口，说明 GUI 正常显示在桌面上", font=("Microsoft YaHei", 12))
lbl.pack(expand=True)
root.after(4000, root.destroy)
root.mainloop()
