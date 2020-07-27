import data_utils
import properties
import data_visualize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
import os
import sys
import random

BATCHSIZE = 4 #6
NUM_ITERATIONS = -1 #150
NUM_TEST_IMAGES = 20 
NUM_RES_NETS = 6
discriminatorTrainingIterations = 4 #2
num_iterations_until_flip = 3
dynamicUpdateIterations = 60

"""
Load the traces in the specified directory. 
"""
def load_traces(directory) :
    print("Loading traces...")
    traces = data_utils.load_traces(directory)
    
    trace_list = list(traces['Trace Number'])
    return trace_list

"""
Plot a simple histogram of the trace list.
"""
def plot_hist_simple(trace_list) :
    plt.hist(trace_list)
    plt.xlabel('Trace Number')
    plt.ylabel('Occurances')
    path = os.getcwd()
    path = os.path.join(path, 'Plots')
    if not (os.path.exists(path)) :
        os.mkdir(path)
    path = os.path.join(path, "Histogram_Trace_Information.pdf")
    plt.savefig(path)
    plt.show()

"""
Read in the csv_files. Due to RAM limitations, it is currently cut down to 7500 entries, as this seems to be the maximum
amount that can still be handled with my current hardware (16GB RAM, i7-5820K processor). For higher amounts, a memory error occurs
and / or the processing is slowed down DRASTICALLY. The time difference between 6000 and 7500 is already several seconds.

To include the remaining lines, uncomment
#i = list(range(1,#desiredlines to skip))
and
app_details = pd.read_csv(properties.APP_DETAILS_PATH, skiprows=i)
-> this also goes for ui_details = pd.read...
or remove the optional parameter totally.
"""
def read_csv_files() :
    print("Reading CSV files...")
    print("Reading App details...")
    #i = list(range(1, 7500))

    app_details = pd.read_csv(properties.APP_DETAILS_PATH)#, nrows=7500)#skiprows=i)
    print("Reading App Package Names...")
    app_details['App Package Name'] = app_details['App Package Name'].to_string()
    print("Reading Play Store Names...")
    app_details['Play Store Name'] = app_details['Play Store Name'].to_string()
    print("Reading Number of Ratings...")
    app_details['Number of Ratings'] = pd.to_numeric(app_details['Number of Ratings'], errors='coerce')
    len(app_details)

    print("Drop empty spaces...")
    app_details = app_details.dropna()
    print("Read UI details...")
    ui_details = pd.read_csv(properties.UI_DETAILS_PATH)#, nrows=7500)#skiprows=i)
    # app_details.nlargest(10, 'Number of Ratings')
    # app_details_medical = app_details.loc[app_details['Category'] == 'Medical']
    
    return app_details, ui_details

"""
Plot the average rating, category counts and number of downloads as histgram and 2 bar plots respectively.
"""
def plot_csv_information(app_details, ui_details) :
    
    print("Calculating histogram plot over the average rating...")

    data_visualize.plot_hist(app_details['Average Rating'], 24, clip_low=2, title="Histogram - Average Rating")

    counts_category = app_details['Category'].value_counts()
    cat = list(counts_category.index.values)
    categories = [w.replace('&', 'and') for w in cat]
    print("Calculating bar plot over app categories...")
    data_visualize.plot_bar(counts_category, categories, y_label='Counts', title="Bar Plot - App Count per Category")

    counts_downloads = app_details['Number of Downloads'].value_counts()
    cat = list(counts_downloads.index.values)
    print("Calculating bar plot over number of downloads...")
    data_visualize.plot_bar(counts_downloads, cat, y_label='Counts', title="Bar Plot - Download Count per App")

"""
Plot a histogram to show how many apps exist with how many screenshots. 
Optionally, a logarithmic scaling for the y-axis can be applied.
"""
def plot_screenshot_information(app_list, scale=None, printScreenshotList=False) :
    count_list = []
    indices = []
    for i in range(1,max(app_list)+1) :
        indices.append(i)
        count_list.append(app_list.count(i))
    
    num_screenshots = 0
    for i in range(len(count_list)) :
        num_screenshots += (count_list[i] * (i+1))
    avg = num_screenshots / len(app_list)
    
    if(printScreenshotList) :
        print("Number of Screenshots: \t Number of Apps:")
        for i in range(len(count_list)) :
            print(str(indices[i])+"\t \t \t "+str(count_list[i]))
        
    print("Total number of screenshots: "+str(num_screenshots))
    
    if (scale == None) :
       data_visualize.plot_bar(count_list, indices, x_label="Number of available screenshots in an app", y_label="Number of apps with this amount of screenshots",
                            title="Number of screenshots per app - on average "+str(round(avg, 2)))#, scale='log') 
    elif (scale == 'log') :
        data_visualize.plot_bar(count_list, indices, x_label="Number of available screenshots in an app", y_label="Number of apps with this amount of screenshots",
                                title="Number of screenshots per app - on average "+str(round(avg, 2)), scale='log')
    else :
        print("Unknown scale for plot. Please use either None or log.")
        
"""
This method takes two parameters:
    - the 64-dimensional vector to find similar vectors to
    - the list of 64-dimensional vectors to search through
This method uses the simple approach of identifying vectors with (as many as possible) matching dimensions that are also 0.
There are two lists returned:
    - the vectors that are supposedly similar, at least sharing the same 0-dimensions
    - a list of indicies so the vectors can be identified again and e.g. the corresponding label extracted from ui_names_list
    
A first simple test showed that when searching for zero-dimensions, the results are already better than when searching for
non-zero dimensions. Thus this simple approach only looks for zero-dimensions, and, depending on checking both ways or not,
ensures that either both vectors have the exact same zero-dimensions or at least as many as the vector to compare with.
"""
def find_vectors_with_similar_zero_entries(vector_to_compare, ui_vectors_list, checkBothDirections=True) :
    if (len(ui_vectors_list) <= 1) :
        print("List of vectors to search through is too short")
        return [], []
    if (len(vector_to_compare) != 64) :
        print("Expected 64-dimensional vector, but vector has only "+str(len(vector_to_compare))+" dimensions.")
        return [], []
    
    indices = []
    similar_vectors = []
    
    for i in range(0, len(ui_vectors_list)) :
        v = ui_vectors_list[i]
        foundEqualZeroDimensions = True
        for j in range(0, len(vector_to_compare)) :
            if (vector_to_compare[j] == 0) :
                if (v[j] == 0) :
                    continue
                    # Everything is fine
                else : 
                    foundEqualZeroDimensions = False
                    break
                
            # This can be cancelled out, so that e.g. it is only compared whether other vectors
            # have 0 value in at least those dimensions that the required vector has.
            if (checkBothDirections) :
                if (v[j] == 0) :
                    if (vector_to_compare[j] == 0) :
                        continue
                        # Everything is fine
                    else : 
                        foundEqualZeroDimensions = False
                        break
        if (foundEqualZeroDimensions) :
            indices.append(i)
            similar_vectors.append(ui_vectors_list[i])
            
    return similar_vectors, indices

"""
This method extracts the ui_names, e.g. "28335.png" from a list of ui_names, based on a list of indices
as returned by find_vectors_with_similar_zero_entries().
"""
def extract_ui_names_from_indices_list(ui_names_list, indices) :
    
    ui_names = []
    for i in indices :
        ui_names.append(ui_names_list[i])
        
    return ui_names        

def TSNE(ui_vectors_list, writeResultsToFile=True) :
    
    # TSNE_representation is a list of vectors; the shape of the list is (len(ui_vectors_list), 2)
    #TSNE_representation = data_utils.calculate_TSNE(ui_vectors_list, perplexity=30, learning_rate=200, n_iter=1000)
    per_1 = 30
    lr_1 = 100
    n_1 = 1000
    
    per_2 = 50
    lr_2 = 100
    n_2 = 1000
    
    TSNE_representation = data_utils.calculate_TSNE(ui_vectors_list, perplexity=per_1, learning_rate=lr_1, n_iter=n_1)
    xs = TSNE_representation[:, 0]
    ys = TSNE_representation[:, 1]
    
    if (writeResultsToFile) :
        np.savetxt('TSNE_'+str(per_1)+'-'+str(lr_1)+'-'+str(n_1)+'.out', TSNE_representation, delimiter=',')
        np.savetxt('TSNE_'+str(per_1)+'-'+str(lr_1)+'-'+str(n_1)+'_xs.out', xs, delimiter=',')
        np.savetxt('TSNE_'+str(per_1)+'-'+str(lr_1)+'-'+str(n_1)+'_ys.out', ys, delimiter=',')
        
    plt.scatter(xs, ys)
    plt.title("TSNE of UI Vectors")
    plt.savefig("TSNE_UI_Vectors-"+str(per_1)+"_"+str(lr_1)+"_"+str(n_1)+".png")
    plt.show()
    plt.clf()
    #data_visualize.plot_scatter(xs, ys, title="TSNE of UI Vectors")
    TSNE_representation = data_utils.calculate_TSNE(ui_vectors_list, perplexity=per_2, learning_rate=lr_2, n_iter=n_2)
    xs = TSNE_representation[:, 0]
    ys = TSNE_representation[:, 1]
    
    if (writeResultsToFile) :
        np.savetxt('TSNE_'+str(per_2)+'-'+str(lr_2)+'-'+str(n_2)+'.out', TSNE_representation, delimiter=',')
        np.savetxt('TSNE_'+str(per_2)+'-'+str(lr_2)+'-'+str(n_2)+'_xs.out', xs, delimiter=',')
        np.savetxt('TSNE_'+str(per_2)+'-'+str(lr_2)+'-'+str(n_2)+'_ys.out', ys, delimiter=',')
        
    plt.scatter(xs, ys)
    plt.title("TSNE of UI Vectors")
    plt.savefig("TSNE_UI_Vectors-"+str(per_2)+"_"+str(lr_2)+"_"+str(n_2)+".png")
    plt.show()
    plt.clf()
    
"""
Find the nth occurance of a given value in a list
"""
def get_nth_index(data, value, n) :
    occurances = (index for index, val in enumerate(data) if val == value)
    return next(islice(occurances, n-1, n), None)    

"""
Parameters:
    v: the index of the vector in the dataset to get nearest neighbors of (same index as in e.g. ui_vectors_list)
    k: the amount of nearest neighbors to obtain
    per: perplexity, see 'file' below
    lr: learning rate, see 'file' below
    n: number of iterations, see 'file' below

'File': The input file has the following format: TSNE_$perplexity$-$learning_rate$-$n_iter$.out

This method calculates the k nearest neighbors of a vector and returns their indices. These indices can the be used 
e.g. in the ui_names_list to obtain the file name of supposedly similar screenshots
"""
def getKNearestNeighbors(v, k, per, lr, n) :
    data = np.loadtxt('TSNE_'+str(per)+'-'+str(lr)+'-'+str(n)+'.out', delimiter=',')
    vector = data[v]
    distances = [np.linalg.norm(vector - data[i]) for i in range(len(data))]
    
    if (k > len(distances)) :
        k = len(distances)
    
    sorted_distances = distances.copy()
    sorted_distances.sort()
    print(sorted_distances[0:30])
    indices = []
    for i in range(k) :
        index = distances.index(sorted_distances[i])
        if (indices.count(index) > 0) :
            index = get_nth_index(distances, sorted_distances[i], indices.count(index)+1)
            # For the rare occassion, that there are three nearest neighbors with the exact same distance, search for the third occurance
            if (indices.count(index) > 0) :
                index = get_nth_index(distances, sorted_distances[i], indices.count(index)+2)
                # For the super rare occassion of 4 equidistant nearest neighbors, search for the 4th occurance
                if (indices.count(index) > 0) :
                    index = get_nth_index(distances, sorted_distances[i], indices.count(index)+3)
                # This could technically be repeated further times, but the current test runs didn't show 4 equidistant items once
        indices.append(index)
        
    return indices

def visualize_general_dataset_information(scale_type=0) :
    #trace_list = load_traces(properties.TRACES_DIR)
    #plot_hist_simple(trace_list)
    #del trace_list
            
    app_details, ui_details = read_csv_files()
    plot_csv_information(app_details, ui_details)
    del app_details, ui_details
    
    app_list = data_utils.count_screenshots(properties.TRACES_DIR)
    print("Length of app list: "+str(len(app_list)))
    
    # Use one of these two, depending on the desired scale of the y-axis
    if (scale_type == 0) :
        plot_screenshot_information(app_list)
    elif (scale_type == 1) :
        plot_screenshot_information(app_list, scale='log')
    else :
        print("Unknown scale type. Please set 0 or 1 or implement further y-axis scalings.")
        
def calculate_TSNE() :
    ui_names_list = data_utils.load_ui_names(properties.VECTORS_DIR, printExcerpt=True)
    ui_vectors_list = data_utils.load_ui_vectors(properties.VECTORS_DIR, printExcerpt=True)
    print("Length of UI Names: "+str(len(ui_names_list)))
    print("Number of vectors: "+str(len(ui_vectors_list)))
    
    print(ui_vectors_list[0][1]==0)
    similar_vectors, indices = find_vectors_with_similar_zero_entries(ui_vectors_list[1], ui_vectors_list)
    
    print("Number of similar vectors: "+str(len(similar_vectors)))
    if (len(similar_vectors) > 1) :
        print(similar_vectors[1])
        print(indices)
        print(extract_ui_names_from_indices_list(ui_names_list, indices))
     
    TSNE(ui_vectors_list)#,writeResultsToFile=False)
    
    indices = getKNearestNeighbors(3, 30, 30, 100, 1000)
    if (len(similar_vectors) > 1) :
        print(similar_vectors[1])
        print(indices)
        print(extract_ui_names_from_indices_list(ui_names_list, indices))
        
def print_excerpt_of_sketch_ui_mapping(paths_sketches, paths_uis, ui_mapping, max_range=5) :
    print("Excerpt of the mapping:")
    for i in range(max_range) :
        print(paths_sketches[ui_mapping[0][i]], paths_uis[ui_mapping[1][i]])
    
    #data_utils.show_images(ui_mapping, paths_uis, paths_sketches, num_images=3)
    
    img = images[0]
    patches = data_utils.get_image_patches(img, 256, crop=True)
    print("Number of Patches = "+str(len(patches)))
    
    for i in range(max_range) :
        plt.subplot(2, max_range, 1+i)
        plt.axis('off')
        plt.imshow(patches[i])
    for i in range(max_range) :
        plt.subplot(2, max_range, 5+i)
        plt.axis('off')
        plt.imshow(patches[i+20])

def print_usage() :
    print("\nUsage:\n"
            +"python "+sys.argv[0]+" <command line argument 1> <command line argument 2> etc. (Alternatively 'python3 ...')\n"
            +"The following command line arguments exist (order does not matter):\n"
            +"-h / help / -help: Show help contents.\n"
            +"-useLabels=x: Specifiy whether to use the CycleGAN model with or without labels (default: True).\n"
            +"-convertGrayscaleToRGB: Check all sketches and uis as defined in properties.py and convert any grayscale to RGB. Requires to restart "+sys.argv[0]+" after execution.\n"
            +"-e / equalizeImageShapes / -equalizeImageShapes: Execute data_utils.equalize_image_shapes(); script ends after equalizing if set.\n"
            +"-r / resize / -resize: Resize all images to 256x256 pixels in the '_small' directories. Use -imageSize to specify the new size (default: 256).\n"
            +"-removeArtifacts: Change all pixel values of sketches to either black or white, so that no intermediate grayscale levels remain.\n"
            +"-resizeSketches: Resize the sketches as well; otherwise only the unique uis will be resized.\n"
            +"-visualizeData: Visualize dataset information. Script terminates after plot creations.\n"
            +"-s / skipTraining / -skipTraining: Do not execute GAN training.\n"
            +"-b=x / batchsize=x / -batchsize=x: Set the batchsize hyperparameter (default = "+str(BATCHSIZE)+").\n"
            +"-i=x / iterations=x / -iterations=x: Set the number of iterations (default = "+str(NUM_ITERATIONS)+").\n"
            +"-resnet=x / resnet=x: Set the number of residual networks in the generator model (default = "+str(NUM_RES_NETS)+").\n"
            #+"-all / allPaths / -allPaths: Include all of Rico's directories, not just the unique_uis directory.\n"
            +"-c / create / -create: Create all GAN models, e.g. to restart with fresh models.\n"
            +"-save / save: Save all newly created models; happens automatically if no models exist yet.\n"
            +"-small / small: Use small (256x256 pixels) images instead of standard sized images. Use also to specify folders resized to other values.\n"
            +"-targetDirectory=x: Specify a target directory where to save models, plots and logs. Make sure to use this parameter before e.g. -clearResultsFile.\n"
            +"-clearResultsFile: If set, the file TargetDirectory/Logs/Results.txt will be cleared (deleted and recreated).\n"
            +"-discriminatorTrainingIterations=x: Set how often the discriminator is trained; x=2 means every second, x=3 means every third iteration etc. (default = "+str(discriminatorTrainingIterations)+").\n"
            +"-flipLabelIterations=x: Set how often labels should be flipped when training the discriminator (depends on -discriminatorTrainingIterations); "
            +"e.g. 3 means every 3rd time the discriminator is trained, labels are flipped (default = "+str(num_iterations_until_flip)+").\n"
            +"-evaluate: Perform an evaluation of gathered data, e.g. calculated losses. Evaluation plots are stored to the 'Plots' directory.\n"
            #+"-strategy=x: For test purposes, change strategy to 0-8\n"
            #+"-distributeImages=x: For test purposes, change distribute_images to True/False or 1/0\n"
            +"-dynamicUpdateIterations=x: Specifies how often discriminator and generator losses are compared to increase or decrease discriminator training speed; a multiple of 60 is recommended.\n"
            +"-maxDiscriminatorTrainingIterations=x: Specifies the maximum number of discriminator training iterations between generator updates (default: 10).\n"
            +"-lossFunction=x: Specify the used loss function, without apostrophs. Choose from 'mse', 'mae', 'bce', 'kld' and 'custom'. Default is 'mse'.\n"
            +"-n_start=x: Specify an epoch index where the training starts (default: 0). Supports lists. Intended for continued training of models for correct folder names etc.\n"
            +"-n_epochs=x: Specify the number of training epochs (default: 150). Supports lists.\n"
            +"-newDiscriminators=x: Specify whether new discriminator models should be created (default: False). Supports lists.\n"
            +"-discriminatorUpdateStopEpoch=x: Specify after which epoch the discriminator is not trained anymore (default: Never). Supports lists.\n"
            +"-discriminatorUpdateContinueEpoch=x: Specify after which epoch the discriminator is trained again (default: Never). Supports lists.\n"
            +"-use_lr_decay=x: Specify whether decaying learning rate after 50 epochs shall be used (default: True)\n"
            +"-lr_decay_factor=x: Specify the learning rate decay factor (default: 0.5)\n"
            +"-delay_lr_decay=x: Specify how long the decay should be delayed. E.g. 100 indicates that lr updates begin after epoch 100. Values <1 cancel this parameter (default: 0)\n"
            +"-lr_decay_frequency=x: Specify how often the learning rate decay updates are applied, e.g. 1 = after every epoch, 50 = after every 50 epochs (default: 50).\n"
            +"-lr_start=x: Specifiy the initial learning rate (default: 0.0002)\n"
            +"-use_composite_prediction=x: Specify whether to use composite models for prediction (default: True)\n"
            +"-sketch_subset=x: Specify whether a subset (1-4) of the sketch dataset should be used (default: All are used).\n"
            +"-t / test / -test: Test the gan training; executed after training, so you might want to skip training using the '-s' parameter.\n"
            +"-trainImages=x: Set the number of test images (default: all).\n"
            +"-testImages=x: Set the number of test images (default = "+str(NUM_TEST_IMAGES)+").\n"
            +"-imageSize=x: Specify the image size (for noise images and resizing images); default: 256.\n"
            +"-shuffleTestImages=x: Specify whether test images should be shuffled before test translations (default: False).\n"
            +"-trainableDiscriminator: Set discriminator.trainable = True in composite models.\n"
            +"-trainableGenerator: Set generator2.trainable = True in composite models.\n"
            +"-matchImages=x: Specify whether to match sketches with their corresponding UIs for training (default: True).\n"
            +"-noise=x: Specify whether noise should be used instead of sketches (default: False).\n"
            +"-simplifyJson: Simplify the json files containing annotations for the dataset.\n"
            +"-one_hot=x: Specify whether to use one-hot encoded labels (default: False).\n"
            +"\n"
            #+"Example call for testing: python "+sys.argv[0]+" -c -save -iterations=3 -t -small\n"
            +"Example call: python "+sys.argv[0]+" -c -save -small -imageSize=64 -test -b=24 -targetDirectory=2019_12_31")

if __name__ == '__main__':
    
    bool_use_labels = True
    
    if (len(sys.argv) > 1) :
        for i in range(1, len(sys.argv)) :
            if ("-useLabels" in sys.argv[i]) :
                bool_use_labels = sys.argv[i].split("=")[1] == "True"
                break
            
    if (bool_use_labels) :
        print("Using the CycleGAN model WITH labels...")
        import cycle_gan_extended as gan
    else :
        print("Using the CycleGAN model WITHOUT labels...")
        import cycle_gan as gan
    
    # A bunch of variables that can be modified through command line arguments
    # There are more modifiable variables located in gan.py
    bool_equalize_image_shapes = False
    bool_resize_images = False
    bool_resize_sketches = False
    bool_remove_artifacts = False
    bool_skip_training = False
    bool_use_small_images = False
    bool_get_all_paths = False
    bool_test_gan = False
    bool_evaluate = False
    bool_convert_grayscale_to_RGB = False
    bool_shuffle_test_set = False
    bool_trainable_discriminator = False
    bool_trainable_generator2 = False
    sketch_subset_index = 0
    n_train_images = 0
    n_epochs = 150
    bool_new_discriminator = False
    discriminator_update_stop_epoch=-1
    discriminator_update_continue_epoch=-1
    bool_match_images = True
    bool_use_noise = False
    image_size = 256
    
    loss_function = 'mse'
    target_directory = ""
    n_start = 0
    
    create_models = False
    save_models = False
    
    # Check for command line arguments
    if (len(sys.argv) > 1) :
        for i in range(1, len(sys.argv)) :
            if (sys.argv[i] == "-h" or sys.argv[i] == "help" or sys.argv[i] == "-help") :
                print_usage()
                sys.exit(0)
            elif ("-useLabels" in sys.argv[i]) :
                continue
            elif (sys.argv[i] == "-convertGrayscaleToRGB") :
                bool_convert_grayscale_to_RGB = True
            elif (sys.argv[i] == "-e" or sys.argv[i] == "equalizeImageShapes" or sys.argv[i] == "-equalizeImageShapes") :
                bool_equalize_image_shapes = True
            elif (sys.argv[i] == "-r" or sys.argv[i] == "resize" or sys.argv[i] == "-resize") :
                bool_resize_images = True
            elif (sys.argv[i] == "-removeArtifacts") :
                bool_remove_artifacts = True
            elif (sys.argv[i] == "-s" or sys.argv[i] == "skipTraining" or sys.argv[i] == "-skipTraining") :
                print("Set skipTraining = True")
                bool_skip_training = True
            elif ("-b=" in sys.argv[i] or "batchsize=" in sys.argv[i]) :
                BATCHSIZE = int(sys.argv[i].split("=")[1])
                print("Set batchsize = "+str(BATCHSIZE))
            elif ("-i=" in sys.argv[i] or "iterations=" in sys.argv[i]) :
                NUM_ITERATIONS = int(sys.argv[i].split("=")[1])
                print("Set num_iterations = "+str(NUM_ITERATIONS))
            elif ("resnet=" in sys.argv[i]) :
                NUM_RES_NETS = int(sys.argv[i].split("=")[1])
                print("Set num_resnets = "+str(NUM_RES_NETS))
            elif (sys.argv[i] == "-all" or sys.argv[i] == "allPaths" or sys.argv[i] == "-allPaths") :
                bool_get_all_paths = True
            elif (sys.argv[i] == "-c" or sys.argv[i] == "create" or sys.argv[i] == "-create") :
                print("Set createNewModels = True")
                create_models = True
            elif (sys.argv[i] == "-save" or sys.argv[i] == "save") :
                print("Set saveNewModels = True")
                save_models = True
            elif (sys.argv[i] == "-small" or sys.argv[i] == "small") :
                print("Using the small 256x256 pixel images.")
                bool_use_small_images = True
            elif (sys.argv[i] == "-resizeSketches") :
                bool_resize_sketches = True
            elif ("-discriminatorTrainingIterations=" in sys.argv[i]) :
                if (int(sys.argv[i].split("=")[1]) >= 1) :
                    discriminatorTrainingIterations = int(sys.argv[i].split("=")[1])
                    print("Set discriminator training iterations to every "+str(discriminatorTrainingIterations)+" iterations.")
            elif ("-flipLabelIterations=" in sys.argv[i]) :
                num_iterations_until_flip = int(sys.argv[i].split("=")[1])
                print("Flipping labels every "+str(num_iterations_until_flip)+" iterations during discriminator training (effectively every "+str(num_iterations_until_flip * discriminatorTrainingIterations)+" iterations).")
            elif ("-dynamicUpdateIterations=" in sys.argv[i]) :
                dynamicUpdateIterations = int(sys.argv[i].split("=")[1])
                print("Set dynamicUpdateIterations to "+str(dynamicUpdateIterations))
            elif ("-sketch_subset=" in sys.argv[i]) :
                sketch_subset_index = int(sys.argv[i].split("=")[1])
                if (sketch_subset_index < 0 or sketch_subset_index > 4) :
                    print("Sketch subset "+str(sketch_subset_index)+" is invalid. Using the default value.")
                else :
                    print("Using only sketch subset "+str(sketch_subset_index))
            elif ("-maxDiscriminatorTrainingIterations=" in sys.argv[i]) :
                gan.set_max_discriminator_training_iterations(int(sys.argv[i].split("=")[1]))
            elif ("-targetDirectory=" in sys.argv[i]) :
                target_directory = sys.argv[i].split("=")[1]
                gan.set_target_directory(target_directory)
            elif ("-lossFunction=" in sys.argv[i]) :
                new_loss_function = sys.argv[i].split("=")[1]
                if (new_loss_function == 'mse' or new_loss_function == 'mean_squared_error') :
                    loss_function = 'mse'
                elif (new_loss_function == 'mae' or new_loss_function == 'mean_absolute_error') :
                    loss_function = 'mae'
                elif (new_loss_function == 'bce' or new_loss_function == 'binary_crossentropy') :
                    loss_function = 'binary_crossentropy'
                elif (new_loss_function == 'kld' or new_loss_function == 'kullback_leibler_divergence') :
                    loss_function = 'kullback_leibler_divergence'
                elif (new_loss_function == 'custom') :
                    loss_function = 'custom'
                else :
                    print("Unknown or not yet implemented loss function '"+new_loss_function+"'. Please use 'mse' (default), 'mae' or 'custom'.")
                    sys.exit(0)
            elif (sys.argv[i] == "-clearResultsFile") :
                if (len(target_directory) > 0) :
                    path = os.path.join(os.getcwd(), target_directory)
                    if not (os.path.exists(path)) :
                        os.mkdir(path)
                    path = os.path.join(path, 'Logs')
                else :
                    path = os.path.join(os.getcwd(), "Logs")
                path = os.path.join(path, "Results.txt")
                if (os.path.exists(path)) :
                    #input("You're about to clear (delete) the file "+path+". This step cannot be undone if executed. "
                    #      +"An empty file is recreated as soon as the GAN training starts.\n"
                    #      +"Press Enter to continue or press CTRL+C to cancel.")
                    os.remove(path)
            elif (sys.argv[i] == "-evaluate") :
                print("Performing an evaluation at the end of this script.")
                bool_evaluate = True
            elif ("-strategy=" in sys.argv[i]) :
                gan.set_strategy(int(sys.argv[i].split("=")[1]))
            elif ("-distributeImages=" in sys.argv[i]) :
                gan.set_distribute_images(sys.argv[i].split("=")[1] == "True")
            elif ("-n_start=" in sys.argv[i]) :
                tmp = sys.argv[i].split("=")[1]
                # If the input is a list, that is e.g. [1,2,3] or (1,2,3) or [1,2,3), remove the first and last symbol
                if ("[" in tmp or "(" in tmp) :
                    tmp = tmp[1:len(tmp) - 1]
                    n_start = [int(tmp.split(",")[i]) for i in range(len(tmp.split(",")))]
                    print("Executing training "+str(len(n_start))+" times with n_start="+str(n_start))
                    del tmp
                elif (int(sys.argv[i].split("=")[1]) >= 0) :
                    n_start = int(sys.argv[i].split("=")[1])
                    print("Epoch count starts at "+str(n_start))
            elif ("-n_epochs=" in sys.argv[i]) :
                tmp = sys.argv[i].split("=")[1]
                # If the input is a list, that is e.g. [1,2,3] or (1,2,3) or [1,2,3), remove the first and last symbol
                if ("[" in tmp or "(" in tmp) :
                    tmp = tmp[1:len(tmp) - 1]
                    n_epochs = [int(tmp.split(",")[i]) for i in range(len(tmp.split(",")))]
                    print("Executing training "+str(len(n_epochs))+" times with n_epochs="+str(n_epochs))
                    del tmp
                elif (int(sys.argv[i].split("=")[1]) >= 0) :
                    n_epochs = int(sys.argv[i].split("=")[1])
                    print("Set number of training epochs to "+str(n_epochs))
            elif ("-newDiscriminators=" in sys.argv[i]) :
                tmp = sys.argv[i].split("=")[1]
                # If the input is a list, that is e.g. [1,2,3] or (1,2,3) or [1,2,3), remove the first and last symbol
                if ("[" in tmp or "(" in tmp) :
                    tmp = tmp[1:len(tmp) - 1]
                    bool_new_discriminator = tmp.split(",")
                    print("Executing training with bool_new_discriminator="+str(bool_new_discriminator))
                    del tmp
                elif (int(sys.argv[i].split("=")[1]) >= 0) :
                    bool_new_discriminator = sys.argv[i].split("=")[1]
                    print("Set creating of new discriminators to "+str(bool_new_discriminator))
            elif ("-discriminatorUpdateStopEpoch=" in sys.argv[i]) :
                tmp = sys.argv[i].split("=")[1]
                # If the input is a list, that is e.g. [1,2,3] or (1,2,3) or [1,2,3), remove the first and last symbol
                if ("[" in tmp or "(" in tmp) :
                    tmp = tmp[1:len(tmp) - 1]
                    discriminator_update_stop_epoch = [int(tmp.split(",")[i]) for i in range(len(tmp.split(",")))]
                    print("During training, stopping discriminator training after epochs "+str(discriminator_update_stop_epoch))
                    del tmp
                elif (int(sys.argv[i].split("=")[1]) >= 0) :
                    discriminator_update_stop_epoch = int(sys.argv[i].split("=")[1])
                    print("Stopping discriminator training after epoch "+str(discriminator_update_stop_epoch))
            elif ("-discriminatorUpdateContinueEpoch=" in sys.argv[i]) :
                tmp = sys.argv[i].split("=")[1]
                # If the input is a list, that is e.g. [1,2,3] or (1,2,3) or [1,2,3), remove the first and last symbol
                if ("[" in tmp or "(" in tmp) :
                    tmp = tmp[1:len(tmp) - 1]
                    discriminator_update_continue_epoch = [int(tmp.split(",")[i]) for i in range(len(tmp.split(",")))]
                    print("During training, continuing discriminator training after epochs "+str(len(discriminator_update_continue_epoch)))
                    del tmp
                elif (int(sys.argv[i].split("=")[1]) >= 0) :
                    discriminator_update_continue_epoch = int(sys.argv[i].split("=")[1])
                    print("Continuing discriminator training after epoch "+str(discriminator_update_continue_epoch))
            elif ("-use_lr_decay=" in sys.argv[i]) :
                gan.set_use_decaying_lr(sys.argv[i].split("=")[1] == "True")
            elif ("-lr_decay_factor=" in sys.argv[i]) :
                gan.set_lr_decay_factor(float(sys.argv[i].split("=")[1]))
            elif ("-lr_decay_frequency=" in sys.argv[i]) :
                gan.set_learning_rate_decay_frequency(int(sys.argv[i].split("=")[1]))
            elif ("-delay_lr_decay=" in sys.argv[i]) :
                gan.set_delayed_decay(int(sys.argv[i].split("=")[1]))
            elif ("-lr_start=" in sys.argv[i]) :
                gan.set_initial_lr(float(sys.argv[i].split("=")[1]))
            elif ("-use_composite_prediction=" in sys.argv[i]) :
                gan.set_use_composite_prediction(sys.argv[i].split("=")[1] == "True")
            elif ("-trainableDiscriminator" == sys.argv[i]) :
                bool_trainable_discriminator = True
            elif ("-trainableGenerator" == sys.argv[i]) :
                bool_trainable_generator2 = True
            elif (sys.argv[i] == "-t" or sys.argv[i] == "test" or sys.argv[i] == "-test") :
                print("Calculating test translations (after training if not skipped).")
                bool_test_gan = True
            elif ("-trainImages=" in sys.argv[i]) :
                n_train_images = int(sys.argv[i].split("=")[1])
                if (n_train_images < 1) :
                    n_train_images = 0
                print("Using "+str(n_train_images)+" images for training (0 = all).")
            elif ("-testImages=" in sys.argv[i]) :
                NUM_TEST_IMAGES = int(sys.argv[i].split("=")[1])
                print("Set number of test images = "+str(NUM_TEST_IMAGES))
            elif ("-imageSize=" in sys.argv[i]) :
                image_size = int(sys.argv[i].split("=")[1])
                if (image_size > 3) :
                    print("Changed image size to "+str(image_size))
                else :
                    print("Image size "+str(image_size)+" is too small. Using default size of 256.")
                    image_size = 256
            elif ("-shuffleTestSet=" in sys.argv[i]) :
                bool_shuffle_test_set = (sys.argv[i].split("=")[1] == "True")
                print("Shuffling test images before test translations = "+str(NUM_TEST_IMAGES))
            elif ("-matchImages=" in sys.argv[i]) :
                bool_match_images = (sys.argv[i].split("=")[1] == "True")
                print("Main.py: Matching images = "+str(bool_match_images))
                gan.set_bool_match_images(bool_match_images)
            elif ("-noise=" in sys.argv[i]) :
                bool_use_noise = (sys.argv[i].split("=")[1] == "True")
                print("Using noise instead of sketches = "+str(bool_use_noise))
            elif ("-simplifyJson" in sys.argv[i]) :
                print("Simplifying the json files in "+properties.UNIQUE_UIS_ANNOTATIONS_DIR+"_small")
                file_path = properties.UNIQUE_UIS_ANNOTATIONS_DIR+"_small"
                data_utils.simplify_annotations(file_path)
                print("Finished simplification process. Please restart the script.")
                sys.exit(0)
            elif ("-visualizeData" in sys.argv[i]) :
                visualize_general_dataset_information(scale_type=1)
                sys.exit(0)
            elif ("-one_hot=" in sys.argv[i]) :
                if (bool_use_labels) :
                    gan.set_use_one_hot(sys.argv[i].split("=")[1] == "True")
                else :
                    print("The parameter -one_hot is only available if labels are used!\n")
            else :
                print_usage()
                print("\nERROR: Unknown argument "+sys.argv[i]+" found.")
                sys.exit(0)
    else :
        print_usage()
        #input("\nSince you called "+sys.argv[0]+" without further arguments, press Enter to start"
        #      +" training the GAN with freshly created models, using small images and default parameters (cf. usage above) and"
        #      +" calculating an example image translation plus doing an evaluation.\nPress CTRL+C to cancel.")
        #create_models = True
        #save_models = True
        bool_use_small_images = True
        bool_evaluate = True
        bool_test_gan = True
        if (len(target_directory) > 0) :
            path = os.path.join(os.getcwd(), target_directory)
            if not (os.path.exists(path)) :
                os.mkdir(path)
            path = os.path.join(path, 'Logs')
        else :
            path = os.path.join(os.getcwd(), "Logs")
        path = os.path.join(path, "Results.txt")
        if (os.path.exists(path)) :
            #input("You're about to clear (delete) the file "+path+". This step cannot be undone if executed. "
            #      +"An empty file is recreated as soon as the GAN training starts.\n"
            #      +"Press Enter to continue or press CTRL+C to cancel.")
            os.remove(path)
    
    images = []
    
    if (bool_convert_grayscale_to_RGB) :
        paths_sketches = np.asarray(data_utils.get_image_paths(properties.SKETCHES_DIR))
        paths_sketches_small = np.asarray(data_utils.get_image_paths(properties.SKETCHES_DIR+"_small"))
        paths_sketches_test = np.asarray(data_utils.get_image_paths(properties.SKETCHES_TEST_DIR))
        paths_sketches_test_small = np.asarray(data_utils.get_image_paths(properties.SKETCHES_TEST_DIR+"_small"))
        paths_uis = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_DIR))
        paths_uis_small = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_DIR+"_small"))

        data_utils.convert_grayscale_to_RGB(paths_sketches)
        data_utils.convert_grayscale_to_RGB(paths_sketches_small)
        data_utils.convert_grayscale_to_RGB(paths_sketches_test)
        data_utils.convert_grayscale_to_RGB(paths_sketches_test_small)

        data_utils.convert_grayscale_to_RGB(paths_uis)
        data_utils.convert_grayscale_to_RGB(paths_uis_small)

        print("All images were checked and converted to RGB images if applicable. Please restart the script to use the new image versions.")
        sys.exit(0)
        
    if (bool_remove_artifacts) :
        if (bool_use_small_images) :
            paths_sketches = np.asarray(data_utils.get_image_paths(properties.SKETCHES_DIR+"_small"))
        else :
            paths_sketches = np.asarray(data_utils.get_image_paths(properties.SKETCHES_DIR))
            
        data_utils.remove_artifacts(paths_sketches)
        print("All sketches have their artifacts removed. Please restart the script to use the updated sketches.\n")
        sys.exit(0)

    if (bool_get_all_paths) :
        # Once per code execution, all paths have to be checked out once. This is requires unavoidable 14 seconds
        # By storing the paths in this variable, this process must only be done once instead of searching the paths for each file-read
        paths_full = data_utils.get_image_paths(properties.TRACES_DIR)
        
    if (bool_use_small_images) :
        if (image_size == 256) :
            # Get the paths of the unique uis
            paths_uis = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_DIR+"_small"))
            # Get the paths of the sketches
            paths_sketches = np.asarray(data_utils.get_image_paths(properties.SKETCHES_DIR+"_small"))
        else :
            # Get the paths of the unique uis
            paths_uis = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_DIR+"_small_"+str(image_size)))
            # Get the paths of the sketches
            paths_sketches = np.asarray(data_utils.get_image_paths(properties.SKETCHES_DIR+"_small_"+str(image_size)))
    else :
        # Get the paths of the unique uis
        paths_uis = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_DIR))
        # Get the paths of the sketches
        paths_sketches = np.asarray(data_utils.get_image_paths(properties.SKETCHES_DIR))
    
    if (bool_equalize_image_shapes) :
        ui_mapping, _ = data_utils.get_matching_uis(paths_sketches, paths_uis)
        data_utils.equalize_image_shapes(paths_sketches, paths_uis, ui_mapping)
        if not (bool_resize_sketches) :
            print("Image shapes were equalized. Please restart the script to use the updated files.")
            sys.exit(0)
        
    if (bool_resize_images) :
        if (bool_use_small_images and image_size != 256) :
            paths_uis = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_DIR+"_small_"+str(image_size)))
            paths_uis_test = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_TEST_DIR+"_small_"+str(image_size)))
            paths_sketches = np.asarray(data_utils.get_image_paths(properties.SKETCHES_DIR+"_small_"+str(image_size)))
            paths_sketches_test = np.asarray(data_utils.get_image_paths(properties.SKETCHES_TEST_DIR+"_small_"+str(image_size)))
        elif not (bool_use_small_images) :
            paths_sketches = np.asarray(data_utils.get_image_paths(properties.SKETCHES_DIR+"_small"))
            paths_sketches_test = np.asarray(data_utils.get_image_paths(properties.SKETCHES_TEST_DIR+"_small"))
            paths_uis = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_DIR+"_small"))
            paths_uis_test = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_TEST_DIR+"_small"))
        else :
            paths_sketches = np.asarray(data_utils.get_image_paths(properties.SKETCHES_DIR))
            paths_sketches_test = np.asarray(data_utils.get_image_paths(properties.SKETCHES_TEST_DIR))
            paths_uis = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_DIR))
            paths_uis_test = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_TEST_DIR))
        
        if (bool_resize_sketches) :
            if (len(paths_sketches) > 0) :
                print("Calling resize_images() with new image size "+str(image_size)+" for "+properties.SKETCHES_DIR+"_small ...")
                data_utils.resize_images(paths_sketches, image_size, image_size)
            if (len(paths_sketches_test) > 0) :
                print("Calling resize_images() with new image size "+str(image_size)+" for "+properties.SKETCHES_TEST_DIR+"_small ...")
                data_utils.resize_images(paths_sketches_test, image_size, image_size)
        if (len(paths_uis) > 0) :
            print("Calling resize_images() with new image size "+str(image_size)+" for "+properties.UNIQUE_UIS_DIR+"_small ...")
            data_utils.resize_images(paths_uis, image_size, image_size)
        if (len(paths_uis_test) > 0) :
            print("Calling resize_images() with new image size "+str(image_size)+" for "+properties.UNIQUE_UIS_TEST_DIR+"_small ...")
            data_utils.resize_images(paths_uis_test, image_size, image_size)
            
        print("Images were resized. Please restart the script to use the updated files.")
        sys.exit(0)

    # Create a mapping of indices from sketch list and uis list
    # The ui_mapping contains a mapping of sketches to uis, and 
    # sketches_without_ui contains all sketches that could not be mapped (maybe useful for testing?)
    # Otherwise drop them by calling: 'ui_mapping, _ = data_utils.get_matching_uis()'    
    if (bool_match_images) :
        path_mapping = os.path.join(os.getcwd(), 'Logs')
        path_mapping = os.path.join(path_mapping, 'Mapping.txt')
        if not (os.path.exists(path_mapping)) :
            ui_mapping, _ = data_utils.get_matching_uis(paths_sketches, paths_uis)
            paths_sketches = [paths_sketches[ui_mapping[0][i]] for i in range(len(ui_mapping[0]))]
            paths_uis = [paths_uis[ui_mapping[1][i]] for i in range(len(ui_mapping[1]))]
            with open(path_mapping, 'a+') as file :
                for i in range(len(ui_mapping[0])) :
                    file.write(""+paths_sketches[i]+","+paths_uis[i]+"\n")

        else :
            with open(path_mapping, 'r') as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines]
            paths_sketches = [lines[i].split(",")[0] for i in range(len(lines))]
            paths_uis = [lines[i].split(",")[1] for i in range(len(lines))]
    else :
        random.shuffle(paths_uis)
        paths_uis = paths_uis[0:len(paths_sketches)]
            
    # If specified, the training dataset is reduced in size by choosing only the subset that is from a specified designer.
    if (sketch_subset_index > 0) :
        subset_string = "_"+str(sketch_subset_index)
        paths_tmp = [[paths_sketches[i] for i in range(len(paths_sketches)) if subset_string in paths_sketches[i]],
                     [paths_uis[i] for i in range(len(paths_sketches)) if subset_string in paths_sketches[i]]]
        paths_sketches = list(paths_tmp[0])
        paths_uis = list(paths_tmp[1])
        print("Sketch subset "+str(sketch_subset_index)+" contains "+str(len(paths_sketches))+" sketches and "+str(len(paths_uis))+" corresponding UIs.")
        for i in range(min(3, len(paths_sketches))) :
            print(paths_sketches[i], paths_uis[i])
        del paths_tmp
        
    """
    Prepare the labels according to sketches and UIs.
    If not yet done, the mappings are written to disk to 'src/Logs/LabelMapping.txt', independent of a specified target directory
    """
    if (bool_use_labels) :
        sketchLabels = data_utils.get_label_paths(properties.SKETCHES_ANNOTATIONS_DIR+"_small")
        uiLabels = data_utils.get_label_paths(properties.UNIQUE_UIS_ANNOTATIONS_DIR+"_small")
        path_label_mapping = os.path.join(os.getcwd(), 'Logs')
        path_label_mapping = os.path.join(path_label_mapping, 'LabelMapping.txt')
        if not (os.path.exists(path_label_mapping)) :
            paths_sketches_tmp, sketchLabels_tmp, paths_uis_tmp, uiLabels_tmp = [], [], [], []
            for i in range(len(paths_sketches)) :
                sketch = paths_sketches[i]
                sketch = ''.join([s for s in sketch if s.isdigit()])
                for j in range(len(sketchLabels)) :
                    label = sketchLabels[j]
                    label = ''.join([l for l in label if l.isdigit()])
                    if (sketch == label) :
                        paths_sketches_tmp.append(paths_sketches[i])
                        sketchLabels_tmp.append(sketchLabels[j])
                        paths_uis_tmp.append(paths_uis[i])
                        break
            paths_sketches = list(paths_sketches_tmp)
            sketchLabels = list(sketchLabels_tmp)
            paths_uis = list(paths_uis_tmp)
            for i in range(len(paths_uis)) :
                ui = paths_uis[i]
                ui = ''.join([s for s in ui if s.isdigit()])
                for j in range(len(uiLabels)) :
                    label = uiLabels[j]
                    label = ''.join([l for l in label if l.isdigit()])
                    if (ui == label) :
                        uiLabels_tmp.append(uiLabels[j])
                        break
            uiLabels = list(uiLabels_tmp)
            del paths_sketches_tmp, sketchLabels_tmp, paths_uis_tmp, uiLabels_tmp
            
            with open(path_label_mapping, 'a+') as file :
                for i in range(len(paths_sketches)) :
                    file.write(""+paths_sketches[i]+","+sketchLabels[i]+","+paths_uis[i]+","+uiLabels[i]+"\n")
                    
        else :
            with open(path_label_mapping, 'r') as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines]
            paths_sketches = [lines[i].split(",")[0] for i in range(len(lines))]
            sketchLabels = [lines[i].split(",")[1] for i in range(len(lines))]
            paths_uis = [lines[i].split(",")[2] for i in range(len(lines))]
            uiLabels = [lines[i].split(",")[3] for i in range(len(lines))]

    if (bool_use_noise) :
        paths_sketches = np.reshape(np.random.rand(len(paths_sketches) * image_size * image_size * 3), (len(paths_sketches), image_size, image_size, 3))
    else :
        shuffleTrainingSet = list(zip(paths_sketches, paths_uis))
        random.shuffle(shuffleTrainingSet)
        paths_sketches, paths_uis = zip(*shuffleTrainingSet)
        del shuffleTrainingSet
           
    if not (bool_skip_training) :
        print("\nStarting training with:\n"
              +"Number of sketches = "+str(len(paths_sketches))
              +"\nNumber of uis = "+str(len(paths_uis))
              +"\nBatchsize = "+str(BATCHSIZE)
              +"\nLoss function = "+loss_function+"\n")
        
        image_shape = (256,256,3)
        
        if (len(target_directory) > 0) :
            path = os.path.join(os.getcwd(), target_directory)
            if not (os.path.exists(path)) :
                os.mkdir(path)
            path = os.path.join(path, 'Models')
        else :
            path = os.path.join(os.getcwd(), 'Models')
            
        if (create_models) :
            print("Creating fresh models...")
            # generator: A -> B
            g_model_AtoB = gan.define_generator(image_shape, n_resnet=NUM_RES_NETS)#, plotName='generator_modelAtoB_plot.png')
            # generator: B -> A
            g_model_BtoA = gan.define_generator(image_shape, n_resnet=NUM_RES_NETS)#, plotName='generator_modelBtoA_plot.png')
            # discriminator: A -> [real/fake]
            d_model_A = gan.define_discriminator(image_shape, loss_function=loss_function)#, plotName='discriminator_modelAtoB_plot.png')
            # discriminator: B -> [real/fake]
            d_model_B = gan.define_discriminator(image_shape, loss_function=loss_function)#, plotName='discriminator_modelBtoA_plot.png')
            # composite: A -> B -> [real/fake, A]
            c_model_AtoB = gan.define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape,
                                                      loss_function=loss_function, bool_trainable_discriminator=bool_trainable_discriminator, 
                                                      bool_trainable_generator2=bool_trainable_generator2)#, plotName='composite_modelAtoB_plot.png')
            # composite: B -> A -> [real/fake, B]
            c_model_BtoA = gan.define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape, 
                                                      loss_function=loss_function,bool_trainable_discriminator=bool_trainable_discriminator, 
                                                      bool_trainable_generator2=bool_trainable_generator2)#, plotName='composite_modelBtoA_plot.png')
            
            print("Created models.")
            if (save_models) :
                print("Saving models...")
                gan.save_model(g_model_AtoB, "g_model_AtoB")
                gan.save_model(g_model_BtoA, "g_model_BtoA")
                gan.save_model(d_model_A, "d_model_A")
                gan.save_model(d_model_B, "d_model_B")
                gan.save_model(c_model_AtoB, "c_model_AtoB")
                gan.save_model(c_model_BtoA, "c_model_BtoA")
                print("Saved the freshly created models.")

            # After saving the models, free the space. 
            # Since the models are loaded in the training routine anyways, the allocated space can be freed already
            del d_model_A
            del d_model_B
            del g_model_AtoB
            del g_model_BtoA
            del c_model_AtoB
            del c_model_BtoA
            
        if (n_train_images > 0) :
            if (n_train_images == 1) :
                paths_sketches = paths_sketches[0]
                paths_uis = paths_uis[0]
                if (bool_use_labels) :
                    sketchLabels = sketchLabels[0]
                    uiLabels = uiLabels[0]
            else :
                paths_sketches = paths_sketches[0:n_train_images]
                paths_uis = paths_uis[0:n_train_images]
                if (bool_use_labels) :
                    sketchLabels = sketchLabels[0:n_train_images]
                    uiLabels = uiLabels[0:n_train_images]
            
        if (isinstance(n_start, list)) :
            for i in range(len(n_start)) :
                if (len(n_start) != len(n_epochs) or len(n_start) != len(bool_new_discriminator) or len(n_epochs) != len(bool_new_discriminator)) :
                    print("The number of elements of n_start, n_epochs and bool_new_discriminator do not match ("
                          +str(len(n_start))+","+str(len(n_epochs))+","+str(len(bool_new_discriminator))+").\n")
                    sys.exit(1)
                if (isinstance(discriminator_update_stop_epoch, list)) :
                    index = 0
                    for j in range(len(discriminator_update_stop_epoch)) :
                        if (discriminator_update_stop_epoch[j] < n_start[i]) :
                            index = j+1
                        else :
                            break
                    discriminator_update_stop_epoch = discriminator_update_stop_epoch[index:len(discriminator_update_stop_epoch)]
                if (isinstance(discriminator_update_continue_epoch, list)) :
                    index = 0
                    for j in range(len(discriminator_update_continue_epoch)) :
                        if (discriminator_update_continue_epoch[j] < n_start[i]) :
                            index = j+1
                        else :
                            break
                    discriminator_update_continue_epoch = discriminator_update_continue_epoch[index:len(discriminator_update_continue_epoch)]
                    
                if (bool_use_labels) :
                    gan.train(paths_sketches, sketchLabels, paths_uis, uiLabels, batchsize=BATCHSIZE, n_steps=NUM_ITERATIONS, 
                          discriminatorTrainingIterations=discriminatorTrainingIterations, num_iterations_until_flip=num_iterations_until_flip,
                          dynamicUpdateIterations=dynamicUpdateIterations, n_start=n_start[i], n_epochs=n_epochs[i], bool_new_discriminator=(bool_new_discriminator[i]=="True"), 
                          discriminator_update_stop_epoch=discriminator_update_stop_epoch, discriminator_update_continue_epoch=discriminator_update_continue_epoch,
                          image_size=image_size) 
                else :
                    gan.train(paths_sketches, paths_uis, batchsize=BATCHSIZE, n_steps=NUM_ITERATIONS, 
                          discriminatorTrainingIterations=discriminatorTrainingIterations, num_iterations_until_flip=num_iterations_until_flip,
                          dynamicUpdateIterations=dynamicUpdateIterations, n_start=n_start[i], n_epochs=n_epochs[i], bool_new_discriminator=(bool_new_discriminator[i]=="True"), 
                          discriminator_update_stop_epoch=discriminator_update_stop_epoch, discriminator_update_continue_epoch=discriminator_update_continue_epoch,
                          image_size=image_size) 
        else :
            if (bool_use_labels) :
                gan.train(paths_sketches, sketchLabels, paths_uis, uiLabels, batchsize=BATCHSIZE, n_steps=NUM_ITERATIONS, 
                          discriminatorTrainingIterations=discriminatorTrainingIterations, num_iterations_until_flip=num_iterations_until_flip,
                          dynamicUpdateIterations=dynamicUpdateIterations, n_start=n_start, n_epochs=n_epochs, bool_new_discriminator=bool_new_discriminator, 
                          discriminator_update_stop_epoch=discriminator_update_stop_epoch, discriminator_update_continue_epoch=discriminator_update_continue_epoch,
                          image_size=image_size)
            else :
                gan.train(paths_sketches, paths_uis, batchsize=BATCHSIZE, n_steps=NUM_ITERATIONS, 
                          discriminatorTrainingIterations=discriminatorTrainingIterations, num_iterations_until_flip=num_iterations_until_flip,
                          dynamicUpdateIterations=dynamicUpdateIterations, n_start=n_start, n_epochs=n_epochs, bool_new_discriminator=bool_new_discriminator, 
                          discriminator_update_stop_epoch=discriminator_update_stop_epoch, discriminator_update_continue_epoch=discriminator_update_continue_epoch,
                          image_size=image_size) 
        
    if (bool_test_gan) :
        if (NUM_TEST_IMAGES == 20) : 
            print("Preparing test images. Tip: Use the cmd argument '-testImages=x' to customize the number of test images.")
        else :
            print("Preparing test images...")
            
        # Load all test images w.r.t. to the chosen testset
        if (bool_use_small_images) :
            if (image_size == 256) :
                paths_uis_test = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_TEST_DIR+"_small"))
                paths_sketches_test = np.asarray(data_utils.get_image_paths(properties.SKETCHES_TEST_DIR+"_small"))
            else :
                paths_uis_test = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_TEST_DIR+"_small_"+str(image_size)))
                paths_sketches_test = np.asarray(data_utils.get_image_paths(properties.SKETCHES_TEST_DIR+"_small_"+str(image_size)))         
        else :
            paths_uis_test = np.asarray(data_utils.get_image_paths(properties.UNIQUE_UIS_TEST_DIR))
            paths_sketches_test = np.asarray(data_utils.get_image_paths(properties.SKETCHES_TEST_DIR))
            
        if (bool_use_labels) :
            
            # To speed up the label loading process, a mapping is created and written to disk if not yet existent
            path_test_label_mapping = os.path.join(os.getcwd(), 'Logs')
            path_test_label_mapping = os.path.join(path_test_label_mapping, 'TestLabelMapping.txt')
            if not (os.path.exists(path_test_label_mapping)) :
                sketchTestLabels = data_utils.get_label_paths(properties.SKETCHES_TEST_ANNOTATIONS_DIR+"_small")
                
                uiValidationImages = list()
                uiValidationPaths = data_utils.get_image_paths(properties.UNIQUE_UIS_DIR+"_small")
                ui_mapping = data_utils.get_matching_uis_for_testing(sketchTestLabels, uiValidationPaths)
                
                for i in range(len(ui_mapping[1])) :
                    if isinstance(ui_mapping[1][i], int) :
                        uiValidationImages.append(uiValidationPaths[ui_mapping[1][i]])
                    else :
                        uiValidationImages.append(None)
                
                paths_sketches_tmp, sketchLabels_tmp, uiValidationImages_tmp = [], [], []
                for i in range(len(paths_sketches_test)) :
                    sketch = paths_sketches_test[i]
                    sketch = ''.join([s for s in sketch if s.isdigit()])

                    for j in range(len(sketchTestLabels)) :
                        label = sketchTestLabels[j]
                        label = ''.join([l for l in label if l.isdigit()])
                        
                        if (sketch == label) :
                            paths_sketches_tmp.append(paths_sketches_test[i])
                            sketchLabels_tmp.append(sketchTestLabels[j])
                            uiValidationImages_tmp.append(uiValidationImages[j])
                            break

                paths_sketches_test = list(paths_sketches_tmp)
                sketchTestLabels = list(sketchLabels_tmp)
                uiValidationImages = list(uiValidationImages_tmp)
                
                del paths_sketches_tmp, sketchLabels_tmp, uiValidationImages_tmp
                
                with open(path_test_label_mapping, 'a+') as file :
                    for i in range(min(len(paths_sketches_test), len(sketchTestLabels), len(uiValidationImages))) :
                        file.write(""+paths_sketches_test[i]+","+sketchTestLabels[i]+","+str(uiValidationImages[i])+"\n")
                 
            # Otherwise, the mapping is simply read from a file
            else :
                with open(path_test_label_mapping, 'r') as file :
                    lines = [line.strip() for line in file.readlines()]
                    paths_sketches_test, sketchTestLabels, uiValidationImages = [], [], []
                    
                    for i in range(len(lines)) :
                        line = lines[i].split(",")
                        paths_sketches_test.append(line[0])
                        sketchTestLabels.append(line[1])
                        if (line[2] == "None") :
                            uiValidationImages.append(None)
                        else :
                            uiValidationImages.append(line[2])
                    
            uiTestLabels = data_utils.get_label_paths(properties.UNIQUE_UIS_TEST_ANNOTATIONS_DIR+"_small")
                      
            # If test images are used that have a corresponding UI, the images are preloaded for the test_translation() method
            bool_preload_images = False
            for i in range(len(uiValidationImages)) :
                if (uiValidationImages[i] != None) :
                    bool_preload_images = True
                    
            if (bool_preload_images) :
                uiValidationImages_tmp = list()
                for i in range(len(uiValidationImages)) :
                    if (uiValidationImages[i] == None) :
                        uiValidationImages_tmp.append(None)
                    else :
                        uiValidationImages_tmp.append(data_utils.load_image(uiValidationImages[i]))
                uiValidationImages = list(uiValidationImages_tmp)
                del uiValidationImages_tmp
        
            # If desired, the testset can be shuffled before the testdata is cut to the specified size and translations are computed
            if (bool_shuffle_test_set) :
                # Unless one test set is completely empty, both are shuffled so that possible matchings are maintained.
                if (len(paths_sketches_test == paths_uis_test)) :
                    shuffleTestSet = list(zip(paths_sketches_test, sketchTestLabels, uiValidationImages, paths_uis_test, uiTestLabels))
                    random.shuffle(shuffleTrainingSet)
                    paths_sketches_test, sketchTestLabels, uiValidationImages, paths_uis_test, uiTestLabels = zip(*shuffleTestSet)
                    del shuffleTestSet
        else :
            if (bool_shuffle_test_set) :
                # Unless one test set is completely empty, both are shuffled so that possible matchings are maintained.
                if (len(paths_sketches_test == paths_uis_test)) :
                    shuffleTestSet = list(zip(paths_sketches_test, paths_uis_test))
                    random.shuffle(shuffleTrainingSet)
                    paths_sketches_test, paths_uis_test = zip(*shuffleTestSet)
                    del shuffleTestSet
                else :
                    random.shuffle(paths_sketches_test)
                    random.shuffle(paths_uis_test)
				
        # Cut the testset to the specified size NUM_TEST_IMAGES
        paths_sketches_test = paths_sketches_test[0:NUM_TEST_IMAGES]
        paths_uis_test = paths_uis_test[0:NUM_TEST_IMAGES]
        if (bool_use_labels) :
            sketchTestLabels = sketchTestLabels[0:NUM_TEST_IMAGES]
            uiTestLabels = uiTestLabels[0:NUM_TEST_IMAGES]
        
        if (bool_use_noise) :
            paths_sketches_test = np.reshape(np.random.rand(len(paths_sketches_test) * image_size * image_size * 3), (len(paths_sketches_test), image_size, image_size, 3))
        
        # Compute the image translations. If a test set is empty, it is skipped.
        if (len(paths_sketches_test) > 0) :
            print("Calculating test translations for "+str(len(paths_sketches_test))+" from domain A...")
            if (bool_use_labels) :
                if (len(sketchTestLabels) > 0) :
                    gan.test_translation(paths_sketches_test, sketchTestLabels, "g_model_AtoB", "g_model_BtoA", direction="AtoB", UIs=uiValidationImages)
                else :
                    print("Could not find any sketch test labels!")
            else :
                gan.test_translation(paths_sketches_test, "c_model_AtoB", direction="AtoB")
        else :
            if (bool_use_small_images) :
                print("Could not find any test data in "+properties.SKETCHES_TEST_DIR+"_small")
            else :
                print("Could not find any test data in "+properties.SKETCHES_TEST_DIR)
                
        if (len(paths_uis_test) > 0) :
            print("Calculating test translations for "+str(len(paths_uis_test))+" from domain B...")
            if (bool_use_labels) :
                if (len(uiTestLabels) > 0) :
                    gan.test_translation(paths_uis_test, uiTestLabels, "g_model_BtoA", "g_model_AtoB", direction="BtoA", UIs=uiValidationImages)
                else :
                    print("Could not find an UI test labels!")
            else :
                gan.test_translation(paths_uis_test, "c_model_BtoA", direction="BtoA")
        else :
            if (bool_use_small_images) :
                print("Could not find any test data in "+properties.UNIQUE_UIS_TEST_DIR+"_small")
            else :
                print("Could not find any test data in "+properties.UNIQUE_UIS_TEST_DIR)
        
    if (bool_evaluate) :
        if (len(target_directory) > 0) :
            path = os.path.join(os.getcwd(), target_directory)
            if not (os.path.exists(path)) :
                os.mkdir(path)
            path = os.path.join(path, 'Logs')
            savepath = os.path.join(path, 'Plots')
        else :
            path = os.path.join(os.getcwd(), 'Logs')
            savepath = os.path.join(os.getcwd(), 'Plots')
        file = 'Results.txt'
        data_visualize.plot_losses(path, file, savepath)

    """
    ###########################################################################
    
    This section contains various code snippets and examples of currently unused methods
    so that they can easily be inserted into the code at a later state. Quick
    content overview:
        1. Visualizing dataset information
        2. Methods when deploying the code with freshly downloaded datasets
        3. General hardware information
        4. Manual time measurement
        5. Hyperparameter usage
        6. Normalize / Denormalize images
        7. Get existing image shapes in a dataset
        8. Split images into patches and reassemble them
        9. Show patches in a nice way (various subplots)
        10. Read results from file
    
    ###########################################################################
    1. Currently unused methods for obtaining and visualizing dataset information
    
    #visualize_general_dataset_information()
    #calculate_TSNE()
    
    ###########################################################################
    
    2. Currently unused methods, which should be called when deploying the code, as
    these methods ensure that sketches and their corresponding uis have the same size and are all stored as RGB images
    Otherwise, the keras.model.prediction method of the GAN is the latest point where the implementation will fail
    
    #data_utils.equalize_image_shapes(paths_sketches, paths_uis, ui_mapping)
    #data_utils.print_broken_image_paths(paths_sketches)#paths_uis[np.asarray(ui_mapping[1])])
    #data_utils.convert_grayscale_to_RGB(paths_sketches)
    
    ###########################################################################
    
    3. Some general information regarding tests of the GAN with the datasets:
    
    The ui-image folder contains 22.4GB data (132.522 files (66.261) in total) 
    -> roughly 11GB RAM are required if loaded to provide quick access, 
    as these 22.4GB include .json files, which are exactly 50% of the files in the folder
    
    The loading process needs ~ 45 seconds for all images, otherwise 10-35 seconds per batch (of e.g. 500)
    Storing the images in the RAM speeds up the training time immensely, as the 10-35 seconds loading time per batch
    are cancelled, but instead, roughly 11 GB RAM are blocked. Alternatively, this can of course be split into smaller
    parts, e.g. 2 different image blocks with 5.5GB each, 3 blocks with ~ 3.7GB each or 4 block with <3GB each.
    
    The measured time that is sometimes written in a method's description comment refers to my hardware, including
    - 16GB RAM
    - Intel i7-5820K @3.30 GHz with 12 cores
    - Windows 10 64-bit
    
    ###########################################################################
    
    # 4. For manual time measurement:
    time_measurement_start = time.process_time()
    print("Total time taken for <insert action> = " + str(time.process_time() - time_measurement_start) + " seconds.")
    
    ###########################################################################
    
    # 5. Usage example of the BATCHSIZE hyperparameter. This will be part of the overall GAN implementation.
    # Select the desired amount of images from this list, e.g. by using a BATCHSIZE hyperparameter
    start_index = 0
    paths = paths_sketches[start_index:start_index+BATCHSIZE]
    #paths = paths_uis[start_index:start_index+BATCHSIZE]
    
    ###########################################################################
    
    # 6. Use "low=0" to normalize to/from range [0..1] instead of [-1..1], depending on the activation function
    normalized_image = data_utils.normalize_image(image, low=-1)
    denormalized_image = data_utils.denormalize_image(normalized_image, low=-1))
    
    ###########################################################################
    
    # 7. In order to get all existing image shapes in the datasets
    for i in range(len(images_sketches)) :
        if (images_sketches[i].shape not in shapes) :
            shapes.append(images_sketches[i].shape)
    print(shapes)
    
    remaining_start = 0
    for j in range(0, len(paths_uis), 1000) :
        if (j+1000 > len(paths_uis)) :
            remaining_start = j
            break
        images_uis = data_utils.load_images(paths_uis[j:j+1000])
        for i in range(len(images_uis)) :
            if (images_uis[i].shape not in shapes) :
                shapes.append(images_uis[i].shape)
        print(str(j)+": "+str(shapes))
                
    images_uis = data_utils.load_images(paths_uis[remaining_start:len(paths_uis)])
    for i in range(len(images_uis)) :
        if (images_uis[i].shape not in shapes) :
            shapes.append(images_uis[i].shape)
    
    print(shapes)
    # > [(1920, 1080, 3), (960, 540, 3), (1080, 1920, 3), (540, 960, 3)]
    ###########################################################################
    
    # 8. Split an image into patches and assemble the image from its patches
    image_before = data_utils.load_image(paths_uis[0])
    image_after = data_utils.assemble_image_from_patches(data_utils.get_image_patches(image_before, 256))
    
    ###########################################################################
    
    # 9. Show multiple images in a nice way
    show_image_excerpt = True # Set to False if this snippet should be not executed during the method execution
    if (show_image_excerpt) :
        for i in range(5) :
            plt.subplot(2, 5, 1+i)
            plt.axis('off')
            plt.imshow(images_sketches[1+i*2])
        for i in range(5) :
            plt.subplot(2, 5, 6+i)
            plt.axis('off')
            plt.imshow(images_uis[1+i*2])
            
    ###########################################################################
            
    # 10. Read results from file:
    path = os.path.join(os.getcwd(), "Logs")
    path = os.path.join(path, "Results.txt")
    dA_loss, dB_loss, g_loss = data_utils.load_results_from_file(path)
    print("dA_loss: "+str(dA_loss)+"\ndB_loss: "+str(dB_loss)+"\ng_loss: "+str(g_loss))
    
    ###########################################################################
    """
