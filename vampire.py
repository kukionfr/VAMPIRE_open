#!/usr/bin/env python

# interface libraries
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *
# my files
from mainbody import mainbody
from getboundary import getboundary


def makeform(root, fields):
    entries = {}
    rows = []
    for field in fields:
        row = Frame(root)
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        if field == 'Build Model' or field == '' or field == 'Apply Model':
            lab = Label(row, width=30, text=field, anchor='w', font=("Helvetica", 16))
            lab.pack(side=LEFT)
        else:
            ent = Entry(row)
            if field == 'Number of shape modes':
                ent.insert(0, "choose a number")
            elif field == 'Status':
                ent.insert(0, 'welcome to the vampire analysis')
            elif field == 'Name of the model':
                ent.insert(0, 'name your model')
            elif field == 'Image sets for building' or field == 'Image sets for applying':
                ent.insert(0, '<--- click to load csv')
            elif field == 'Number of coordinates':
                ent.insert(0, '50')
            else:
                ent.insert(0, "<--- click to load model")
            ent.pack(side=RIGHT, expand=YES, fill=X)
            entries[field] = ent
            lab = Label(row, width=24, text=field, anchor='w')
            lab.pack(side=LEFT)
        rows.append(row)
    return entries, rows


def getdir(entries, target):
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'searching...')
    #################################################################
    folder = StringVar()
    foldername = filedialog.askdirectory()
    folder.set(foldername)
    folder = folder.get()
    entries[target].delete(0, END)
    entries[target].insert(0, folder)
    #################################################################
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'directory found...')


def getcsv(entries, target):
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'searching...')
    #################################################################
    folder = StringVar()
    foldername = filedialog.askopenfilename()
    folder.set(foldername)
    folder = folder.get()
    entries[target].delete(0, END)
    entries[target].insert(0, folder)
    #################################################################
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'directory found...')


def Model(entries, buildModel, progress_bar):
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'modeling initiated...')
    #################################################################
    coord_num = entries['Number of coordinates'].get()
    # input definition
    if buildModel:
        csv = entries['Image sets for building'].get()
        clnum = entries['Number of shape modes'].get()
        modelname = entries['Name of the model'].get()  # name
        getboundary(csv, progress_bar, entries)  # create registry csv and boundary stack
        mainbody(buildModel, csv, entries, modelname, clnum, progress_bar)
    else:
        csv = entries['Image sets for applying'].get()
        # modelname = None  #path
        modelname = entries['Model to apply'].get()
        # modelch2 = entries['Model for ch2'].get()
        clnum = None
        getboundary(csv, progress_bar, entries)  # create registry csv and boundary stack
        mainbody(buildModel, csv, entries, modelname, clnum, progress_bar)
    #################################################################
    progress_bar["value"] = 100
    progress_bar.update()
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'modeling completed...')

# vampire graphical user interface
def vampire():
    root = Tk()
    root.style = Style()
    root.style.theme_use('clam')
    # background color of GUI does not match the theme color by default
    root.configure(background='#dcdad5')
    root.style.configure("red.Horizontal.TProgressbar", troughcolor='gray', background='#EA6676')
    # title of the GUI
    root.title("Vampire Analysis")
    # content of the GUI
    fields = (
        'Build Model', 'Image sets for building', 'Number of shape modes', 'Number of coordinates', 'Name of the model',
        '',  # build model button
        'Apply Model', 'Image sets for applying', 'Model to apply', '',
        'Status', '')
    ents, rows = makeform(root, fields)
    # add progress bar
    progress_bar = Progressbar(rows[11], style="red.Horizontal.TProgressbar", orient="horizontal", mode="determinate",
                               maximum=100, length=150)
    progress_bar.pack(side=LEFT, padx=5, pady=5)
    # function 1 : select image set CSV
    b1 = Button(rows[1], text='load csv', width=12, command=(lambda e=ents: getcsv(e, 'Image sets for building')))
    b1.pack(side=RIGHT, padx=5, pady=5)
    # function 2 : construct model
    b2 = Button(rows[5], text='build model', width=12, command=(lambda e=ents: Model(e, True, progress_bar)))
    b2.pack(side=RIGHT, padx=5, pady=5)
    # function 3 : select image set CSV
    b3 = Button(rows[7], text='load csv', width=12, command=(lambda e=ents: getcsv(e, 'Image sets for applying')))
    b3.pack(side=RIGHT, padx=5, pady=5)
    # function 4 : select model
    b4_a = Button(rows[8], text='load model', width=12, command=(lambda e=ents: getdir(e, 'Model to apply')))
    b4_a.pack(side=RIGHT, padx=5, pady=5)
    # function 5 : apply model
    b5 = Button(rows[9], text='apply model', width=12, command=(lambda e=ents: Model(e, False, progress_bar)))
    b5.pack(side=RIGHT, padx=5, pady=5)

    root.mainloop()


vampire()