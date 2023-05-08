from tkinter import *
from tkinter import ttk
from screeninfo import get_monitors

def donothing():
    x = 0

root = Tk(className="Diffusion Optimization")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

frm = ttk.Frame(root, padding=10)
frm.grid()

menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="New", command=donothing)
filemenu.add_command(label="Open", command=donothing)
filemenu.add_command(label="Save", command=donothing)
menubar.add_cascade(label="File", menu=filemenu)
root.config(menu=menubar)

ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)

root.mainloop()
