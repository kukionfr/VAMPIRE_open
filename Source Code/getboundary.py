# built-in libraries
import os
from tkinter import END
import subprocess
# external libraries
from PIL import Image
from scipy.ndimage.measurements import center_of_mass, label
from scipy.ndimage import generate_binary_structure
import numpy as np
import cv2
import pandas as pd
from skimage import measure


def createimstack(tag,setfolder):
    ext = ['.tiff', '.tif', '.jpeg', '.jpg', '.png', '.bmp', '.gif']
    imlist = [_ for _ in os.listdir(setfolder) if _.lower().endswith(tuple(ext))]
    imlist = [_ for _ in imlist if tag.lower() in _.lower()]
    imlist = sorted(imlist)
    imlistpath = [os.path.join(setfolder, _) for _ in imlist]
    imstack = [np.array(Image.open(im)) for im in imlistpath]
    return imstack, imlistpath, imlist


def check_label_status(im):
    if len(list(set(im.flatten()))[1:]) > 1:
        return 'labeled'


def mask2boundary(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.empty(2)
    for i in range(len(contours[0])):
        contour = np.vstack((contours[0][i][0], contour))
    contour = contour[0:-1]
    contour.T[0] = contour.T[0] + 1
    contour.T[1] = contour.T[1] + 1
    boundary = np.empty((2, len(contour.T[1])))
    boundary[0] = contour.T[1]
    boundary[1] = contour.T[0]
    boundary = boundary.T.astype(int)
    return boundary


def getboundary(csv, progress_bar, entries):
    print('## getboundary.py')
    ui = pd.read_csv(csv)
    setpaths = ui['set location']
    # iterate through image set
    for setfolderidx, setfolder in enumerate(setpaths):
        tag = ui['tag'][setfolderidx]
        registry = []
        datasheet = 'VAMPIRE datasheet ' + tag + '.csv'
        registry_dst = os.path.join(setfolder, datasheet)
        boundarymaster = []
        boundarydst = os.path.join(setfolder, tag + '_boundary_coordinate_stack.pickle')
        if os.path.exists(registry_dst):
            print('registry or boundary already exist')
            continue
        imstack, imlistpath, imlist = createimstack(tag,setfolder)
        try:
            inputim = check_label_status(imstack[0])  # intensity label in greyscale
        except:
            entries['Status'].delete(0, END)
            entries['Status'].insert(0, 'error: update your CSV file')
            return
        if inputim is not 'labeled':
            s = generate_binary_structure(2, 2)
            imstack = [label(im, structure=s)[0] for im in imstack]
        # iterate through labeled greyscale image
        for imidx, im in enumerate(imstack):
            labels = list(set(im.flatten()))[1:]
            labels = sorted(labels)
            # iterate through labeled object in image
            for objidx, lab in enumerate(labels):
                mask = np.array((im == lab).astype(int), dtype='uint8')
                boundary = mask2boundary(mask)
                if len(boundary) < 5:
                    continue

                centroid = [int(np.around(_, 0)) for _ in center_of_mass(mask)]
                centroid.reverse()  # swap to correct x,y
                prop = measure.regionprops(mask)[0]
                area = prop['area']
                perimeter = prop['perimeter']
                majoraxis = prop['major_axis_length']
                minoraxis = prop['minor_axis_length']
                circularity = 4 * np.pi * area / perimeter ** 2
                try:
                    ar = majoraxis / minoraxis
                except:
                    ar = 0
                props = [area, perimeter, majoraxis, minoraxis, circularity, ar]
                # fronttag = [imlist[imidx], imidx + 1, objidx + 1]
                fronttag = [imlist[imidx], imidx + 1, lab] # Add image object id
                registry_item = fronttag + centroid + props
                registry.append(registry_item)
                boundarymaster.append(boundary)
                progress = 100 * (objidx + 1) / len(labels) / len(imstack) /  \
                           len(setpaths) + 100 * (imidx + 1) / len(imstack) /  \
                           len(setpaths) + 100 * (setfolderidx + 1) / len(setpaths)
                progress_bar["value"] = progress / 2
                progress_bar.update()
        if len(boundarymaster) != len(registry):
            raise Exception('boundary coordinates length does not match registry length')
        if not os.path.exists(boundarydst):
            df = pd.DataFrame(boundarymaster)
            df.to_pickle(boundarydst)
            subprocess.check_call(["attrib", "+H", boundarydst])
        if not os.path.exists(registry_dst):
            df_registry = pd.DataFrame(registry)
            df_registry.columns = ['Filename', 'ImageID', 'ObjectID', 'X', 'Y', 'Area', 'Perimeter',
                                   'Major Axis', 'Minor Axis', 'Circularity', 'Aspect Ratio']
            df_registry.index = df_registry.index + 1
            df_registry.to_csv(os.path.join(setfolder, datasheet), index=False)
    entries['Status'].delete(0, END)
    entries['Status'].insert(0, 'object csv created...')
    return

