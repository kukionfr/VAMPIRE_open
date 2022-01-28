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
            elif field == 'Number of coordinates':
                ent.insert(0, '50')
            elif field == 'Status':
                ent.insert(0, 'welcome to the vampire analysis')
            elif field == 'Model output folder' or field == 'Result output folder':
                ent.insert(0, '<--- click to choose folder to output the model')
            elif field == 'Image sets to build' or field == 'Image sets to apply':
                ent.insert(0, '<--- click to load csv')
            elif field == 'Model to apply':
                ent.insert(0, '<--- click to load pickle file for the model')
            elif field == 'Model name':
                ent.insert(0,'Give a name to your model here')
            else:
                ent.insert(0, "empty")
            ent.pack(side=RIGHT, expand=YES, fill=X)
            entries[field] = ent
            lab = Label(row, width=24, text=field, anchor='w')
            lab.pack(side=LEFT)
        rows.append(row)
    return entries, rows


def getdir(entries, target):
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'searching...')
    folder = StringVar()
    foldername = filedialog.askdirectory()
    folder.set(foldername)
    folder = folder.get()
    entries[target].delete(0, END)
    entries[target].insert(0, folder)
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'directory found...')


def getcsv(entries, target):
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'searching...')
    folder = StringVar()
    foldername = filedialog.askopenfilename()
    folder.set(foldername)
    folder = folder.get()
    entries[target].delete(0, END)
    entries[target].insert(0, folder)
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'directory found...')


def Model(entries, buildModel, progress_bar):
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'modeling initiated...')
    # input definition
    if buildModel:
        csv = entries['Image sets to build'].get()
        clnum = entries['Number of shape modes'].get()
        outpth = entries['Model output folder'].get()  # name
        getboundary(csv, progress_bar, entries)  # create registry csv and boundary stack
        mainbody(buildModel, csv, entries, outpth, clnum, progress_bar)
    else:
        csv = entries['Image sets to apply'].get()
        outpth = entries['Result output folder'].get()
        clnum = None
        getboundary(csv, progress_bar, entries)  # create registry csv and boundary stack
        mainbody(buildModel, csv, entries, outpth, clnum, progress_bar)
    progress_bar["value"] = 100
    progress_bar.update()
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'modeling completed...')

# vampire graphical user interface
def vampire():
    root = Tk()
    root.geometry("520x600")
    root.style = Style()
    root.style.theme_use('clam')
    # background color of GUI does not match the theme color by default
    root.configure(background='#dcdad5')
    # progress bar at the bottom of GUI
    root.style.configure("red.Horizontal.TProgressbar", troughcolor='gray', background='#EA6676')
    # title of the GUI
    root.title("Vampire Analysis")
    # content of the GUI
    fields = (
        'Build Model', 'Image sets to build', 'Number of coordinates', 'Number of shape modes', 'Model output folder',
        'Model name','',  # build model button
        'Apply Model', 'Image sets to apply', 'Model to apply', 'Result output folder', '',
        'Status', '')
    ents, rows = makeform(root, fields)
    # add progress bar
    progress_bar = Progressbar(rows[13], style="red.Horizontal.TProgressbar", orient="horizontal", mode="determinate",
                               maximum=100, length=1000)
    progress_bar.pack(fill=X)
    # function 1 : select image set CSV
    b1 = Button(rows[1], text='load csv', width=12, command=(lambda e=ents: getcsv(e, 'Image sets to build')))
    b1.pack(side=RIGHT, padx=5, pady=5)
    # function 2 : output
    b2 = Button(rows[4], text='choose folder', width=12, command=(lambda e=ents: getdir(e, 'Model output folder')))
    b2.pack(side=RIGHT, padx=5, pady=5)
    # function 2 : construct model
    b3 = Button(rows[6], text='build model', width=12, command=(lambda e=ents: Model(e, True, progress_bar)))
    b3.pack(side=RIGHT, padx=5, pady=5)
    # function 3 : select image set CSV
    b4 = Button(rows[8], text='load csv', width=12, command=(lambda e=ents: getcsv(e, 'Image sets to apply')))
    b4.pack(side=RIGHT, padx=5, pady=5)
    # function 4 : select model
    b5 = Button(rows[9], text='load pickle', width=12, command=(lambda e=ents: getcsv(e, 'Model to apply')))
    b5.pack(side=RIGHT, padx=5, pady=5)
    # function 5 : output
    b6 = Button(rows[10], text='choose folder', width=12, command=(lambda e=ents: getdir(e, 'Result output folder')))
    b6.pack(side=RIGHT, padx=5, pady=5)
    # function 5 : apply model
    b7 = Button(rows[11], text='apply model', width=12, command=(lambda e=ents: Model(e, False, progress_bar)))
    b7.pack(side=RIGHT, padx=5, pady=5)
    root.mainloop()

vampire()


