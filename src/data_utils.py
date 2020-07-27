import os
import json
import re
import pandas as pd
import math
from func_utils import timeit
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.manifold import TSNE
from PIL import Image
import sys

DEBUG = False

"""
If it is desired to apply e.g. @timeit on a function, call this method instead with a condition, e.g. DEBUG.
Depending on the condition, either the method is executed directly or the desired decorator is applied.
"""
def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)
    return decorator

@conditional_decorator(timeit, DEBUG)
def load_traces(file_path):
    traces = pd.DataFrame(columns=('App Package Name', 'Trace Number', 'Gestures'))
    gestures_list = []
    app_name_list = []
    trace_number_list = []
    #num_iterations = 0
    for root, dirs, files in os.walk(file_path):
        """
        if(num_iterations > 200) :
            break
        num_iterations += 1
        """
        for file in files:
            """
            It seems necessary to exclude the '._gestures.json' files, as they include some non-trivial data which cannot be read
            without proper conversion. However, when inspecting these files, they seem related to MacOS.
            Perhaps this works on Unix, but on Windows it does not.
            """
            if file.endswith("gestures.json") and "._" not in file:
                path = os.path.join(root, file)
                app_name = os.path.basename(os.path.normpath(os.path.dirname(root)))
                trace_number_str = os.path.basename(os.path.normpath(root))
                trace_number = re.findall(r'\d+', trace_number_str)[0]
                with open(path) as jsonfile:
                    gestures = json.load(jsonfile)
                    gestures = [(k, v) for k, v in gestures.items()]
                trace_number_list.append(trace_number)
                app_name_list.append(app_name)
                gestures_list.append(gestures)
    traces['App Package Name'] = app_name_list
    traces['Trace Number'] = trace_number_list
    traces['Gestures'] = gestures_list
    return traces

"""
Read results that were written to a file and return them nicely grouped into dA_loss, dB_loss and g_loss.
Please ensure that there is at most one header line (e.g. "Results:") and only one set of results. 
Otherwise this method might either fail or return incorrect results.
Empty and too short lines as well as all lines that include "Results" will automatically be skipped.

Parameters:
    file_path: The complete path, including the filename.
    
Returns:
    Three lists containing all read loss tuples, meaning three lists that contain multiple tuples with 2 elements each.
"""
@conditional_decorator(timeit, DEBUG)
def load_results_from_file(file_path) :
    dA_loss, dB_loss, g_loss = [], [], []
    with open(file_path) as file :
        # Exclude linebreaks
        lines = [line.rstrip() for line in file]
        lines_to_remove = []
        for i in range(len(lines)) :
            # If there is a header line as created by default, exclude the header line OR
            # If the line has not enough content, e.g. an empty line, skip the line
            if ("Results" in lines[i] or len(lines[i]) <= 10 or "Total time taken" in lines[i]) :
                lines_to_remove.append(i)
        # Remove all invalid lines (effectively skipping them)
        for i in range(len(lines_to_remove)-1, -1, -1) :
            index = lines_to_remove[i]
            lines = lines[0:index]+lines[index+1:len(lines)]

        start_indices = []
        stop_indices = []
        # Get the indices where to find the three tuples in each line
        for line in lines :
            start_indices.append([pos for pos, char in enumerate(line) if char == "["])
            stop_indices.append([pos for pos, char in enumerate(line) if char == "]"])
        # Read the tuples from each line and append them to the result lists
        for i in range(len(start_indices)) :
            result_tuple = (lines[i][start_indices[i][0]+1: stop_indices[i][0]]).split(",")
            dA_loss.append([float(result_tuple[0]), float(result_tuple[1])])
            result_tuple = (lines[i][start_indices[i][1]+1: stop_indices[i][1]]).split(",")
            dB_loss.append([float(result_tuple[0]), float(result_tuple[1])])
            result_tuple = (lines[i][start_indices[i][2]+1: stop_indices[i][2]]).split(",")
            g_loss.append([float(result_tuple[0]), float(result_tuple[1])])
            
    return dA_loss, dB_loss, g_loss

"""
Load a list of images in a randomized fascion.
Use num_random_numbers to specify the amount of images to load

Moved from gan.py to here, as it appeared multiple times there

Requires roughly:
    - 40-45 seconds for all UIs at once
    - 10-34 seconds for a batch of 500 images
"""
@conditional_decorator(timeit, DEBUG)
def load_images_randomized(paths, num_random_numbers) :
    # Choose random indices so that images can occasionally be reused during training
    indices = np.random.randint(0, len(paths), num_random_numbers)
    
    # Load the corresponding images
    paths = [paths[indices[i]] for i in range(len(indices))]
    images = [load_image(paths[i]) for i in range(len(paths))]
    
    return images

def load_image(path) :
    return plt.imread(path)

@conditional_decorator(timeit, DEBUG)
def load_images(paths) :
    # If only one element is submitted as parameter
    if (isinstance(paths, str)) :
        print(paths)
        return np.asarray([plt.imread(paths)])
   
    results = list()
    for i in range(len(paths)) :
        try :
            results.append(plt.imread(paths[i]))
        # Appearantly, without catching the ValueError exception,
        # it can happen that an exception is raised when loading an image
        # With the same data, nothing is printed when these lines are added...
        except ValueError :
            print("Failure on reading image "+str(paths[i]))
            continue
        
    return np.asarray(results)
    #return np.asarray([plt.imread(paths[i]) for i in range(len(paths))])

@conditional_decorator(timeit, DEBUG)
def parallelize_imread(paths) :
    # Parallelize the reading of images. Due to this, the necessary time to read all 70,000 images is massively reduced
    # E.g. on my machine to roughly 6-9 seconds for 500 images, in total ~ 14-21 minutes for all images instead of multiple hours!
    # However, this might not be necessary on more powerful machines, especially if they provide SSDs, as I'm using an HDD
    try :
        from joblib import Parallel, delayed
    except Exception :
        print("\n###################\nModule joblib not found. Cannot execute method parallelize_imread()")
        print("\nPlease note, that joblib does NOT work if keras_contrib is (partially) imported!\n")
        print("Installation is possible via e.g.\n"+
              "- pip: pip install joblib\n"+
              "- anaconda: conda install -c anaconda joblib"+
              "\n####################")
        return []
    
    try :
        num_cores = multiprocessing.cpu_count()
        #print("Number of available cores: "+str(num_cores))
        pool = multiprocessing.Pool()
            
        # Images is a list that contains all images that were read. paths must be an iterable
        images = Parallel(n_jobs=num_cores, backend='threading')(delayed(load_image)(path) for path in paths)
    
        pool.close()
        pool.join()
    except Exception :
        print("parallelize_imread(): Joblib does not work if keras_contrib is (partially) imported. Please use another version of this method.")
        return []
    
    return images

@conditional_decorator(timeit, DEBUG)
def get_image_paths(directory) :
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if (file.endswith(".jpg") or file.endswith(".png")) and "._" not in file:
                paths.append(os.path.join(root, file))
        
    return paths

@conditional_decorator(timeit, DEBUG)
def get_label_paths(directory) :
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if (file.endswith(".json")):
                paths.append(os.path.join(root, file))
        
    return paths

"""
Crop an image by removing pixels from borders until the shape of the image is a multiple of (size x size)
Returns the cropped image

Needs roughly 300-400ms for HD images (1920x1080)
"""
@conditional_decorator(timeit, DEBUG)
def crop_image(image, size) :
    # If the number of rows does not match a multiple of the requested size, crop the image
    rows = image.shape[0]
    if (rows % size != 0) :
        new_size = math.floor(rows / size) * size
        
        # Remove half the rows on the end
        rows_to_remove = math.floor((rows - new_size) / 2)
        print("New size = "+str(new_size)+"\nRows to remove = "+str(rows_to_remove))
        while (rows_to_remove > 0) :
            max_row = image.shape[0]
            image = np.delete(image, max_row - 1, 0)
            rows_to_remove -= 1
        # and the other half on the front
        # as the border are often completely black or white and contain no special information
        rows_to_remove = math.ceil((rows - new_size) / 2)
        print("New size = "+str(new_size)+"\nRows to remove = "+str(rows_to_remove))
        while (rows_to_remove > 0) :
            image = np.delete(image, 0, 0)
            rows_to_remove -= 1
            
    # Repeat the same process for the columns
    cols = image.shape[1]
    if (cols % size != 0) :
        new_size = math.floor(cols / size) * size
        # Remove half the rows on the end
        cols_to_remove = math.floor((cols - new_size) / 2)
        while (cols_to_remove > 0) :
            max_col = image.shape[1]
            image = np.delete(image, max_col - 1, 1)
            cols_to_remove -= 1
        # and the other half on the front
        # as the border are often completely black or white and contain no special information
        cols_to_remove = math.ceil((cols - new_size) / 2)
        while (cols_to_remove > 0) :
            image = np.delete(image, 0, 1)
            cols_to_remove -= 1
            
    return image

"""
Assembles an image given its patches. Currently implemented for images with shapes:
    Width x Height
    - 540x960 (512-768x767-1023 due to image cropping in get_image_patches()) - 2x3 patches
    - 1080x1920 (1792-2047x1024-1279 due to image cropping in get_image_patches()) - 4x7 patches
as these are the only shapes that currently exist in the two used datasets.
If another amount of patches is given, a general approach of assembly is taken, though it might lead to 
the image not being assembled correctly or some patches being excluded from the assembly

Parameters:
    - image patches (as a list, the patches are aligned automatically)
    
Returns:
    - the assembled image
    
TODO: Currently, no interpolation is applied between the patches. Thus 'reassembled' predicted images might have 
obvious borders between patch transitions.
"""
def assemble_image_from_patches(patches) :
    # General checks that appropriate data is 
    if not (isinstance(patches, list)) :
        print("Patches must be a list of images.")
        return np.zeros((256, 256))
    elif (len(patches) == 1) :
        return patches[0]
    elif (len(patches) == 2) :
        return np.vstack((patches[0], patches[1]))
    elif (len(patches) in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31)) :
        print("Found a prime amount of patches. Thus the original image cannot be assembled.")
        return patches[0]
    
    # 540x960 / 960x540, or more general: 512-768x767-1023
    if (len(patches) == 6) :
        image = np.hstack((patches[0], patches[1]))
        for i in range(1,3) :
            image = np.vstack((image, (np.hstack((patches[2*i], patches[2*i+1])))))
    # 1080x1920 / 1920x1080, or more general: 1792-2047x1024-1279
    elif (len(patches) == 28) :
        image = np.hstack((patches[0], patches[1], patches[2], patches[3]))
        for i in range(1,7) :
            image = np.vstack((image, (np.hstack((patches[4*i], patches[4*i+1], patches[4*i+2], patches[4*i+3])))))
    
    # Otherwise the image size is not yet implemented, so a general approach is taken.  
    else :
        print("For the given number of "+str(len(patches))+" patches, the assemble() method is not yet implemented.\n"
              +"Hence, a general approach is taken, which might lead to an incorrect layout or some patches not being included.")
        num_rows, num_columns = 2, 2
        if (len(patches) % 3 == 0) :
            num_rows = 3
            num_columns = len(patches) / 3
        elif (len(patches) % 5 == 0) :
            num_rows = 5
            num_columns = len(patches) / 5
        elif (len(patches) % 7 == 0) :
            num_rows = 7
            num_columns = len(patches) / 7
        elif (len(patches) % 7 == 0) :
            num_rows = 11
            num_columns = len(patches) / 11
        # Prepare the basis to append other patches to
        image = np.hstack((patches[0], patches[1]))
        # Finish the first row
        for i in range(2, num_columns) :
            image = np.hstack((image, patches[i]))
        # Iterating over rows, assemble the image per row and append new rows to the previously assembled rows
        for j in range(1, num_rows) :
            # Get the first patch of a new row
            img_tmp = patches[num_columns+j*num_columns]
            # Iterate over the columns of this row
            for i in range(1, num_columns) :
                # Append new patches within the row to the already assembled row
                img_tmp = np.hstack((img_tmp, patches[num_columns+j*num_columns+i]))
            # Append the just assembled row on the bottom of the so far assembled image
            image.vstack((image, img_tmp))
        
            
    return image

"""
For a given image, returns a list of quadratic patches with size (size X size)
If the image dimensions are not multiples of size, the image is cropped at the borders so that
the most likely unimportant pixels are cut off to avoid losing important information

The cropping is included by index shifting, avoiding the usage of the cropping method
reducing the required time to <= 1ms

Parameters:
    image: the input image to get patches of
    size: the desired size; used for both length and width of the patch, thus the result has size (size x size)
    crop: (Boolean) Toggle whether the image should be cropped if the dimensions are not a multiplier of size in a 'smart' way
          However, if set to False and the dimensions are not a multiplier of size, the remaining rows / columns are cropped
"""
#@conditional_decorator(timeit, DEBUG)
def get_image_patches(image, size, crop=False) :
    image = np.asarray(image)
    patches = []
    # Before getting patches, check whether the image has to be cropped so that
    # no pixels are left out of the patches. Instead, parts of borders are removed as these 
    # are usually without special information.
    # As an example, a 1920x1080 image is cropped down to 1792x1024 for size=256(x256)
    rows = image.shape[0]
    cols = image.shape[1]
    
    if (crop) :
        if (rows % size != 0 or cols % size != 0) :
            new_size_rows = math.floor(rows / size) * size
            new_size_cols = math.floor(cols / size) * size
            num_rows_to_skip_start = math.floor((rows - new_size_rows) / 2)
            num_rows_to_skip_end = math.ceil((rows - new_size_rows) / 2)
            num_cols_to_skip_start = math.floor((cols - new_size_cols) / 2)
            num_cols_to_skip_end = math.ceil((cols - new_size_cols) / 2)
            # Iterate over the patches (i = rows, j = cols), s.t. patches are 
            # upper leftmost patch, 2nd upper left, 3rd upper left etc. until
            # 3rd bottom right patch, 2nd bottom right, bottom rightmost patch 
            for i in range(num_rows_to_skip_start, image.shape[0], size) :
                if (i + size > rows - num_rows_to_skip_end) :
                    break
                for j in range(num_cols_to_skip_start, image.shape[1], size) :
                    if (j + size > cols - num_cols_to_skip_end) :
                        break
                    patch = image[i:i+size, j:j+size]
                    patches.append(patch)
        else :
            # Iterate over the patches (i = rows, j = cols), s.t. patches are 
            # upper leftmost patch, 2nd upper left, 3rd upper left etc. until
            # 3rd bottom right patch, 2nd bottom right, bottom rightmost patch 
            for i in range(0, image.shape[0], size) :
                for j in range(0, image.shape[1], size) :
                    patch = image[i:i+size, j:j+size]
                    patches.append(patch)
    else :
        # Iterate over the patches (i = rows, j = cols), s.t. patches are 
        # upper leftmost patch, 2nd upper left, 3rd upper left etc. until
        # 3rd bottom right patch, 2nd bottom right, bottom rightmost patch 
        for i in range(0, image.shape[0], size) :
            if (i + size > rows) :
                break
            for j in range(0, image.shape[1], size) :
                if (j + size > cols) :
                    break
                patch = image[i:i+size, j:j+size]
                patches.append(patch)

    return patches
            
def get_image_patches_from_image_list(images, size, crop=False) :
    # Get image patches for every single image
    results = list()
    for image in images:
        results.append(list(get_image_patches(image, size, crop=crop)))
    # Flatten the result list, as otherwise the output is a list containing num_images sublists with their individual
    # number of patches each
    result = list()
    for sublist in results :
        for item in sublist :
            result.append(item)
    return np.asarray(result)

"""
This method can be used to shrink large images to a smaller size.
More precisely, the original datasets Rico and hand-drawn sketches contain images where corresponding images do not
necessary share the same dimensions, e.g. 1080x1920 against 540x960.
Since CycleGANs in this implementation intend to use image patches, it is important that a patch of 256x256 pixels
in domain A corresponds to the same patch of 256x256 pixels in domain B, instead of being a part of 
an actually 512x512 pixels sized patch 
"""
def equalize_image_shapes(paths_sketches, paths_uis, ui_mapping) :
    #input("You're about to permanently resize images. Press Enter to Continue or CTRL+C to cancel.")
    for i in range(len(ui_mapping[0])) :
        # Open Sketch
        imageA = Image.open(paths_sketches[ui_mapping[0][i]])
        # Open UI
        imageB = Image.open(paths_uis[ui_mapping[1][i]])
        
        # If no matching exists, continue to the next image
        # This happens if a sketch exists, but the corresponding UI does not
        if (imageB == -1) :
            continue
    
        shapeA = imageA.size
        shapeB = imageB.size
        
        if (shapeA[0] % 256 > shapeB[0] % 256 or shapeA[1] % 256 > shapeB[1] % 256) :
            # Resize A to the size of B
            newSize = (shapeB[0], shapeB[1])
            imageA = imageA.resize(newSize, Image.ANTIALIAS)
            imageA.save(paths_sketches[ui_mapping[0][i]])
            
        elif (shapeB[0] % 256 > shapeA[0] % 256 or shapeB[1] % 256 > shapeA[1] % 256) :
            # Resize B to the size of A
            newSize = (shapeA[0], shapeA[1])
            imageB = imageB.resize(newSize, Image.ANTIALIAS)
            imageB.save(paths_uis[ui_mapping[1][i]])

"""
Resize a set of given images to new_width x new_height and replace them on disk.
Note: 
    - This method replaces the images on disk, hence it is an idempotent operation if called twice with
      the same parameters for width and height. This is to save disk space, as the UI images alone require 11GB.
    - The PIL / Pillow library uses the format (width, height), hence this parameter order.

Parameters:
    paths: a list of paths to the images that shall be resized
    new_width: the desired new width to resize the images to
    new_height: the desired new height to resize the images to
    
Returns:
    Nothing. Instead, all resized images are written to disk and will replace the images
    in the specified paths.
"""
def resize_images(paths, new_width, new_height) :
    #input("You're about to permanently resize "+str(len(paths))+" images to the new size of "
    #      +str(new_width)+"x"+str(new_height)+". Press Enter to Continue or CTRL+C to cancel.")
    index = 0
    for path in paths:
        image = Image.open(path)
        image = image.resize((new_width, new_height), Image.ANTIALIAS)
        image.save(path)
        index += 1
        if (index % 500 == 0) :
            print("Resized "+str(index)+" of "+str(len(paths))+" images...")
    print("Resized "+str(index)+" of "+str(len(paths))+" images.")

"""
The hand-drawn sketches are MOSTLY, but not all of them, grayscale images instead of RGB images.
However, as keras' predict() method requires RGB images, call this method with the necessary image path beforehand
to convert all grayscale images within the respective directory to RGB images.

This is an idempotent method!

Requires the module opencv to execute properly, but, as this module is not required for further code execution,
it is imported within this method's scope to avoid installation issues.
Installation is possible via e.g.
- pip: pip install opencv-python
- anaconda: conda install -c menpo opencv
"""
def convert_grayscale_to_RGB(paths, show_image_path=False):
    
    try :
        import cv2
    except Exception :
        print("\n###################\nModule cv2 not found. Cannot execute method convert_grayscale_to_RGB()")
        print("Installation is possible via e.g.\n"+
              "- pip: pip install opencv-python\n"+
              "- anaconda: conda install -c menpo opencv"+
              "\n####################")
        return

    num_images = 0
    for path in paths:
        image = np.array(Image.open(path))
        
        shape = image.shape
        
        if len(shape) < 3 :
            if (show_image_path) :
                print("Converting grayscale image "+path+" to RGB.")
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image, 'RGB')
            image.save(path)
            num_images += 1
    print("Converted "+str(num_images)+" from grayscale to RGB images.")

def print_broken_image_paths(paths):
    index = 0
    for path in paths :
        index += 1
        if (index % 1000 == 0) :
            print("Iteration "+str(index)+"...")
        image = load_image(path)
        patches = get_image_patches(image, 256)
        try :
            patches[0].reshape((1, 256, 256, 3))
        except ValueError:
            print("Broken image path: "+path)

"""
In order to provide useful training data, a mapping of sketches to UIs from Rico is required
Thus, this method does the following:
    - for each sketch, lookup the sketchname in the uis list
    - if the name appears in the uis list, a matching is found and the indices are stored in a result list
    - if there is no such ui with corresponding name, set the mapping to -1
Return:
    - a list of two lists (sketch indices and ui indices)
    - a list of sketches where the corresponding uis do not exist (in the best case, this list is empty and can be discarded)
        These could for instance be used as a test set of images
"""
@conditional_decorator(timeit, DEBUG)
def get_matching_uis(sketches, uis) :
    results = []
    results.append([])
    results.append([])
    
    sketches_without_ui = []
    num_sketches_without_ui = 0
    
    for i in range(len(sketches)) :
        sketch = sketches[i]

        if ('Linux' in sys.platform or 'linux' in sys.platform) :
            ui_name = sketch[sketch.rfind('/'):] # 'path/3556_2.png' -> /3556_2.png' # the '/' ensures that only the ui '3556.png' is found, not e.g. '13356.png'
        else :
            ui_name = sketch[sketch.rfind('\\'):] # 'path\\3556_2.png' -> \\3556_2.png' # the '\\' ensures that only the ui '3556.png' is found, not e.g. '13356.png'
        index = ui_name.index('_')
        ui_name = ui_name[:index]+ui_name[index+2:]
        
        #print(sketch, ui_name)
        matching = -1
        for j in range(len(uis)) :
            #print(uis[j])
            if ui_name in uis[j] :
                matching = j
                break
        #matching = [ui for ui in uis if ui_name in ui]
        #print(matching)
        if (matching == -1) :
            num_sketches_without_ui += 1
            sketches_without_ui.append(sketches[i])
            print("Sketch without existing UI: "+ui_name)
        else :
            results[0].append(i)
            results[1].append(matching)
        
    print("Number of sketches without an existing UI in the uis list: "+str(num_sketches_without_ui))
    return results, sketches_without_ui

def get_matching_uis_for_testing(sketches, uis) :
    results = []
    results.append([])
    results.append([])
    
    for i in range(len(sketches)) :
        sketch = sketches[i]

        if ('Linux' in sys.platform or 'linux' in sys.platform) :
            ui_name = sketch[sketch.rfind('/'):] # 'path/3556_2.png' -> /3556_2.png' # the '/' ensures that only the ui '3556.png' is found, not e.g. '13356.png'
        else :
            ui_name = sketch[sketch.rfind('\\'):] # 'path\\3556_2.png' -> \\3556_2.png' # the '\\' ensures that only the ui '3556.png' is found, not e.g. '13356.png'
        index = ui_name.index('_')
        ui_name = ui_name[:index]+ui_name[index+2:]
        ui_name = ui_name.replace(".json", ".jpg")
        
        #print(sketch, ui_name)
        matching = -1
        for j in range(len(uis)) :
            #print(uis[j])
            if ui_name in uis[j] :
                matching = j
                break
        #matching = [ui for ui in uis if ui_name in ui]
        #print(matching)
        if (matching == -1) :
            results[0].append(i)
            results[1].append(None)
        else :
            results[0].append(i)
            results[1].append(matching)

    return results

"""
Take an input image and normalize the color values from 0..255 to 0..1 or -1..1
Use the parameter 'low' to decide whether to normalize to 0..1 or to -1..1; setting it to anything but 0 sets the latter choice
Depending on the activation, either can be useful. E.g. tanh returns [-1..1] and sigmoid returns [0..1]

Necessary time: approx. 20ms for 1920x1080 images
"""
def normalize_image(img, low=0) :
    image = np.asarray(img)
    image = image.astype('float32')
    image /= 255.0
    if (low != 0) :
        image *= 2
        image -= 1
    return image

"""
Revert the image normalization.
By specifying low != 0, the range [-1..1] is reverted, otherwise [0..1]
"""
def denormalize_image(img, low=0) :
    image = np.asarray(img)
    image = image.astype('float32')
    if (low != 0) :
        image += 1
        image /= 2.0
    #image *= 255.0
    return image

"""
This simple method creates a set of plots that show the sketches in the upper line and corresponding uis in the bottom line.
Use the num_images parameter to specify how many images should be shown, starting at index 0.
If a mapping is used that does not exist in the UIs folder (because only ~90% of all screenshots are available), 
a blank placeholder is inserted.
"""
def show_images(image_mapping, paths_uis, paths_sketches, num_images=3) :
    #paths_sketches[image_mapping[0][i]], paths_uis[image_mapping[1][i]]
    
    # Show the sketches in the upper line
    for i in range(num_images) :
        plt.subplot(2, num_images, 1+i)
        plt.axis('off')
        plt.imshow(plt.imread(paths_sketches[image_mapping[0][i]]).astype('uint8'))

    # Show the corresponding uis in the bottom line
    for i in range(num_images) :
        plt.subplot(2, num_images, 1+num_images+i)
        plt.axis('off')
        if (image_mapping[1][i] != -1) :
            plt.imshow(plt.imread(paths_uis[image_mapping[1][i]]).astype('uint8'))

def count_screenshots(file_path):
    app_list = []
    
    for root, dirs, files in os.walk(file_path):
        if (len(files) > 0) :
            if (files[0].endswith(".jpg")) :
                i = 0
                for file in files:
                    if ("._" not in file) :
                        i += 1
                app_list.append(i)
    #    if (len(files) > 100) :
    #        print(files)
    return app_list

def load_ui_names(file_path, printExcerpt=False):
    ui_names_list = []
    with open(file_path+"\\ui_names.json") as file:
        ui_names_list = json.load(file)
    ui_names_list = [(k, v) for k, v in ui_names_list.items()][0]
    
    if (printExcerpt) :
        print(str(ui_names_list[0])+": "+str(ui_names_list[1][0:3])+", ...")
    return ui_names_list[1]
    #print(ui_names_list[0])
    
def load_ui_vectors(file_path, printExcerpt=False):
    ui_vectors_list = []
    ui_vectors_list = np.load(file_path+"\\ui_vectors.npy")
    if (printExcerpt) :
        print("64 dimensional vector for the first element:\n"+str(ui_vectors_list[0]))
    return ui_vectors_list

"""
For TSNE, the following parameters can / should be adapted to receive better results:
    perplexity: 5-50, default: 30 - the number of nearest neighbors to consider for calculations
    learning_rate: 10-1000, default: 200 - if too small, data is represent as a small dense ball, or equidistant neighbors for each point if too large
    n_iter: 250+, default: 1000 - number of iterations for optimization
    A numpy.array is required as an input.
The result of this method is a n_components dimensional representation of the input data for visualization 
"""
@conditional_decorator(timeit, DEBUG)
def calculate_TSNE(ui_vectors, perplexity=30, learning_rate=200, n_iter=1000) :
    data_embedded = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter).fit_transform(ui_vectors)
    return data_embedded

"""
Read annotation files and store them in a simplified form of [x1,y1,x2,y2,componentLabel] (per line)
"""
def simplify_annotations(file_path) :
    """
    The labels are stored in .json files and have the following order:
        "bounds": [
        x1,
        y1,
        x2,
        y2
        ],
        <possibly unnecessary lines>
        "componentLabel": "Type", where "Type" is e.g. "List Item" or "Icon" or "Text View"
    The component labels are mapped in the global variable 'label_dictionary' to specific numbers (one number is associated to each component type)
    According to http://ranjithakumar.net/resources/mobile-semantics.pdf, there are 24 unique component types
        
    Some components have children with the same bounds, they should be skipped -> hence a counter is needed to remove componentLabels without previous bounds
    
    The indices must be mapped to a 256x256 matrix, then the corresponding entries are replaced with the associated number
    """
    files = []
    skippedFiles = []
    for root, dirs, f in os.walk(file_path) :
        files = f
        print("Starting to simplify the "+str(len(files))+" json files...")
        
    for index in range(len(files)) :
        
        file = os.path.join(file_path, files[index])
        #print("Processing "+str(file))
        with open(file, 'r', encoding="utf8") as f :
            # This variable counts the bounds that are not followed by an immediate component label.
            # If a component label is found without prior bounds, it is skipped as it would overwrite the children's values
            # As an example, a "View" might contain multiple "Image" children, which are separated by a few pixels -> these borders should be kept
            num_bounds = 0
            num_components = 0
            
            # bounds is a list that contains lists [x1,y1,x2,y2,component_label] (tuples do not support changes of size)
            bounds = []
            
            # Read the json file and remove all '\n'
            try :
                lines_tmp = [line.strip() for line in f.readlines()]
            except UnicodeDecodeError :
                print("With utf8-encoding, found unreadable special character in file "+str(f)+". Skipping line.\n")
                skippedFiles.append(f)
                continue
                        
            # Unmodified json files always start with a curly bracket.
            # Thus a file can be skipped if it has been modified already
            if not ('{' in lines_tmp[0]) :
                continue
            lines = []
            
            for i in range(len(lines_tmp)) :
                if ("componentLabel" in lines_tmp[i]) :
                    if (num_bounds > num_components) :
                        lines.append(lines_tmp[i])
                        num_components += 1
                        
                elif ("bounds" in lines_tmp[i]) :
                    line = lines_tmp[i+1] + lines_tmp[i+2] + lines_tmp[i+3] + lines_tmp[i+4]
                    if (len(lines) > 0) :
                        if not ("componentLabel" in lines[len(lines) - 1]) :
                            lines = lines[:len(lines) - 1]
                            num_bounds -= 1

                    if ("-" in line) :
                        print("Detected a - in file "+str(f)+"; changing value to 0.\n")
                        line_tmp = line.split("-")[0] + "0" 
                        line_tmp2 = line.split("-")[1]
                        line_tmp3 = line_tmp2[line_tmp2.find(","):]
                        if (len(line.split("-")) > 2) :
                            line_tmp4 = "0"
                            line = line_tmp + line_tmp3 + line_tmp4
                        else :
                            line = line_tmp + line_tmp3
                        print("Resulting line = "+str(line))
                        #print("Line = "+str(line)+"\n")
                    lines.append([l for l in line.split(",") if l.isdigit()])
                    num_bounds += 1
                
            # If there is no label available, add an image as label
            if (len(lines) % 2 == 1) :
                lines.append('"componentLabel": "Image"')
                    
            for i in range(0, len(lines), 2) :
                bound = [int(x) for x in lines[i]]
                line = lines[i+1][19:len(lines[i+1]) - 1]
                if (line[len(line) - 1] == '"') :
                    line = line[:len(line) - 1]
                bound.append(line)
                bounds.append(list(bound))
  
            # The label images have a default shape of (1440x2560) pixels, so they need to be broken down really bad
            bool_image_too_large = False
            for i in range(len(bounds)) :
                if (bounds[i][3] > 255) :
                    bool_image_too_large = True
                    break
                
            if (bool_image_too_large) :
                for i in range(len(bounds)) :
                    bounds[i][0] = int(bounds[i][0] / 5.6251)  # Since the values must be in the range [0..255],
                    bounds[i][1] = int(bounds[i][1] / 10.0001) # the divisor is slightly larger (by 0.0001) than necessary to reach 256.0
                    bounds[i][2] = int(bounds[i][2] / 5.6251)  # However, the divisor is small enough to not affect any other values than the maximum number 
                    bounds[i][3] = int(bounds[i][3] / 10.0001)
                    
        os.remove(file)
        with open(file, "a") as f :
            f.writelines("%s\n" % str(bound) for bound in bounds)
        
    print("Skipped "+str(len(skippedFiles))+" files because they contain unreadable elements:\n"+str(skippedFiles))
    
"""
After resizing or recoloring grayscale images, it might happen that certain groups of pixels do not share the same color anymore.
As an example, a group of pixels that are 'black' might become 'dark gray', 'black', 'darker gray'.

This method ensures that all artifacts removed so that the only color values that remain are either [0,0,0] or [255,255,255]
To achieve this, pixel values are compared a threshold of 60. As all RGB values are the same for grayscale images converted to RGB, 
it is checked whether pixel[i][j][0] < 127 -> pixel[i][j][:] = 0 and analogously pixel[i][j][0] >= 127 -> pixel[i][j][:] = 255

Multiple tests have shown that 60 seems to be a reasonable value as there are same gray - dark-gray parts in the sketches that do belong to certain components.

Parameters:
    files: A list of files or a single file to modify
    
Returns:
    Nothing. The files are OVERWRITTEN and stored to disk!
"""
def remove_artifacts(files) :
    if (isinstance(files, str)) :
        files = [files]
        
    for file in files :
        image = load_image(file)
        image = image.copy()

        for i in range(image.shape[0]) :
            for j in range(image.shape[1]) :
                if (image[i][j][0] < 60) :
                    image[i][j] = [0,0,0]
                elif (image[i][j][0] >= 60) :
                    image[i][j] = [255,255,255]
                    
        image = Image.fromarray(image, 'RGB')
        image.save(file)