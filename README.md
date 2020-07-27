# Master Thesis

Generating Realistic UIs From Sketches Using Generative Adversarial Networks.<br>

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development purposes.

Please ensure to clone the repository from: https://github.com/JL-Moll/Master_Thesis.git

The following instructions can be used to set up a python environment on a UNIX machine:

1. Ensure that CUDA (version 10.1) and cuDNN (version 7) are installed correctly
1. Install Tensorflow2, e.g. through pip3 install tensorflow-gpu==2.1.0
2. Install Keras, e.g. through pip3 install Keras
3. Install Keras-contrib, pip3 install git+https://www.github.com/keras-team/keras-contrib.git
4. Install further required packages (see below)
	
### Prerequisites

For a full list of necessary libraries, please refer to the [requirements.txt](https://github.com/JL-Moll/Master_Thesis/blob/master/requirements.txt).

Alternatively, pip3 commands are shown in the following for the required packages:
1. pip3 install pandas (eventually requires to run "pip3 install django" beforehand)
2. pip3 install pydot
3. pip3 install h5py
4. pip3 install numpy
5. pip3 install matplotlib
6. pip3 install -U scikit-learn
7. pip3 install Pillow
8. pip3 install imageio (Optional; only required for create_gif.py)

### Get the data

The Rico dataset is publicly available [here](http://interactionmining.org/rico).
The dataset of hand-drawn sketches is publicly available [here](https://github.com/huang4fstudio/swire).
The sketch labels are available in this Git repository as a ZIP-file.

The datasets are expected to be in a 'traces' directory within the root directory of the cloned repository. As an example, the following structure is possible:
- Master_Thesis > src
- Master_Thesis > traces
- Master_Thesis > README.md
- Master_Thesis > requirements.txt

Depending on the desired approach, the 'traces' subdirectory is expected to contain certain subdirectories:
For general dataset information: <br>
1. ui_layout_vectors

For standard-sized images:

2. sketches <br>
3. sketches_test <br>
4. unique_uis <br>
5. unique_uis_test <br>

For working with downsized images:

6. sketches_small <br>
7. sketches_test_small <br>
8. unique_uis_small <br>
9. unique_uis_test_small <br>

For working with labels:

10. sketches_annotations_small <br>
11. sketches_test_annotations_small <br>
12. unique_uis_annotations_small <br>
13. unique_uis_annotations_small <br>

Optional:

13. filtered_traces <br>
14. filtered_traces_test

## Structure
Brief overview of the python files contained in this archive.<br>

- cycle_gan.py: The CycleGAN implementation that works without labels
- cycle_gan_extended.py: The CycleGAN implementation that works with and requires labels
- data_utils.py: General and commonly used methods for working with data, e.g. loading images, preparing image matchings
- data_visualize.py: General methods for plotting data
- func_utils.py: Used for debugging. Provides the timeit decorator.
- main.py: The main file to start any calculation. Please call 'python(3) main.py -h' for a list of command line arguments.
- properties.py: A summary file for data paths.
- create_gif.py: If desired, the results can be transformed into a gif by specifying the target path in and executing this script.

## Examples
- Call 'python(3) main.py -h' to show a list of command line arguments.
- Call 'python(3) main.py -r -resizeSketches -small' to resize all sketches and UIs in the '_small' directories.
- Call 'python(3) main.py -targetDirectory=Experiment01 -small' to start a model training with default parameters on small (256x256), one-hot encoded labeled images. Results are stored in the specified target directory called 'Experiment01'.
- Call 'python(3) main.py -targetDirectory=Experiment01 - small -skipTraining -test' to compute test image translations using pre-trained models located in the specified directory 'Experiment01' and small (256x256) images with one-hot encoded labels. Alternatively, '-test' can be appended to the previous exemplary call to compute test image translations directly after training the models.

## References 
- Biplab Deka, Zifeng Huang, Chad Franzen, Joshua Hibschman, Daniel Afergan, Yang Li, Jeffrey Nichols and Ranjitha Kumar. 2017. Rico: A Mobile App Dataset for Building Data-Driven Design Applications. In Proceedings of the 30th Annual Symposium on User Interface Software and Technology (UIST '17)
- Thomas F. Liu, Mark Craft, Jason Situ, Ersin Yumer, Radomir Mech, and Ranjitha Kumar. 2018. Learning Design Semantics for Mobile Apps. In The 31st Annual ACM Symposium on User Interface Software and Technology (UIST '18).
- Forrest Huang, John F. Canny, and Jeffrey Nichols. Swire: Sketchbased User Interface Retrieval. ACM ISBN 978-1-4503-5970-2/19/05, 2019. url: https://doi.org/10.1145/3290605.3300334.
- For further references, please refer to the Bibliography of my thesis.
