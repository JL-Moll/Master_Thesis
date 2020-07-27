# Requires to install imageio, e.g. through 'pip3 install imageio' or 'conda isntall -c conda-forge imageio'
try :
    import imageio
except ImportError :
    print("This script requires the module imageio! It can be installed through e.g.:\npip3 install imageio\nconda install -c conda-forge imageio")
    import sys
    sys.exit(1)
import os

if (os.path.exists("L:\\Uni\\Master_Thesis\\Experimente")) :
    datapath = "L:\\Uni\\Master_Thesis\\Experimente"
else :
    # Please specify your datapath here.
    datapath = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.join(datapath, "Experiments")
if not (os.path.exists(datapath)) :
    print("Datapath "+datapath+" not found. Please create and prepare this directory or specify the datapath manually in the code.")
    sys.exit(1)

directories = []
for root, dirs, files in os.walk(datapath) :
    for dir in dirs:
        if ('Epochs_Plots' in dirs) :
            directories.append(os.path.join(root, dir))
if (len(directories) == 0) :
    print("Could not find any valid images. Did you set the correct datapath?\nCurrent datapath = "+datapath)
    sys.exit(1)

for datapath in directories :
    filenamesA = []
    filenamesB = []
    for root, dirs, files in os.walk(datapath) :
        for file in files :
            if ("Example_0" in file) :
                dir = file[0:9]
                filenamesA.append(os.path.join(os.path.join(datapath, dir), file))
            elif ("Translation-" in file) :
                filenamesB.append(os.path.join(datapath, file))

    if (len(filenamesA) > 0) :
        print("Found "+str(len(filenamesA))+" files.")
    elif (len(filenamesB) > 0) :
        print("Found "+str(len(filenamesB))+" files.")

    images = []
    if (len(filenamesA) > 0) :
        for filename in filenamesA:
            images.append(imageio.imread(filename))
        savepath = os.path.abspath(os.path.join(datapath, os.pardir))
        print("Storing gif to path: "+savepath)
        imageio.mimsave(os.path.join(savepath,'AnimatedTrainingResults.gif'), images, fps=2)
    if (len(filenamesB) > 0) :
        for filename in filenamesB:
            images.append(imageio.imread(filename))
        savepath = os.path.abspath(os.path.join(datapath, os.pardir))
        print("Storing gif to path: "+savepath)
        imageio.mimsave(os.path.join(savepath,'AnimatedTestResults.gif'), images, fps=1)