import os

# DIRECTORY PATHS
if (os.path.exists("L:\\Uni\\Master_Thesis\\traces")) :
    # Try to set the absolute path due to my individualized file organisation
    ROOT_DIR = "L:\\Uni\\Master_Thesis\\traces" #os.path.dirname(os.path.abspath(__file__))
else:
    # If executed on another machine, use the default path
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    
DATA_DIR = os.path.join(os.path.dirname(ROOT_DIR), 'traces')
SKETCHES_DIR = os.path.join(DATA_DIR, 'sketches')
TRACES_DIR = os.path.join(DATA_DIR, 'filtered_traces')
UNIQUE_UIS_DIR = os.path.join(DATA_DIR, 'unique_uis')
TRACES_TEST_DIR = os.path.join(DATA_DIR, 'filtered_traces_test')
SKETCHES_TEST_DIR = os.path.join(DATA_DIR, 'sketches_test')
UNIQUE_UIS_TEST_DIR = os.path.join(DATA_DIR, 'unique_uis_test')
VECTORS_DIR = os.path.join(DATA_DIR, 'ui_layout_vectors')
SKETCHES_ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'sketches_annotations')
SKETCHES_TEST_ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'sketches_test_annotations')
UNIQUE_UIS_ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'unique_uis_annotations')
UNIQUE_UIS_TEST_ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'unique_uis_test_annotations')

APPLES_DIR = os.path.join(DATA_DIR, 'apples_train')
ORANGES_DIR = os.path.join(DATA_DIR, 'oranges_train')
APPLES_TEST_DIR = os.path.join(DATA_DIR, 'apples_test')
ORANGES_TEST_DIR = os.path.join(DATA_DIR, 'oranges_test')

"""
If there are directories named e.g. 'unique_uis_small', they can be included instead of
the default directories by toggling this use_small_images boolean to True.

These directories are intended to contain images of e.g. size 256x256, so that
no patches have to be generated, but instead the image is used as a whole.
"""
use_small_images = False
if (use_small_images) :
    SKETCHES_DIR += "_small"
    UNIQUE_UIS_DIR += "_small"
    SKETCHES_TEST_DIR += "_small"
    UNIQUE_UIS_TEST_DIR += "_small"
    
print("\nUsing the following directories:\n"
      +"SKETCHES_DIR = "+SKETCHES_DIR
      +"\nSKETCHES_TEST_DIR = "+SKETCHES_TEST_DIR
      +"\nUNIQUE_UIS_DIR = "+UNIQUE_UIS_DIR
      +"\nUNIQUE_UIS_TEST_DIR = "+UNIQUE_UIS_TEST_DIR
      +"\n")

# FILE PATHS
APP_DETAILS_PATH = os.path.join(DATA_DIR, 'app_details.csv')
UI_DETAILS_PATH = os.path.join(DATA_DIR, 'ui_details.csv')

def get_label_dictionary(BOOL_USE_ONE_HOT, num_classes=25) :
    if (BOOL_USE_ONE_HOT) :
        label_dictionary = {'Advertisement' : 1,
                        'Background Image' : 2,
                        'Bottom Navigation' : 3,
                        'Button Bar' : 4,
                        'Card' : 5,
                        'Checkbox' : 6,
                        'Drawer' : 7,
                        'Date Picker' : 8,
                        'Image' : 9,
                        'Image Button' : 10,
                        'Input' : 11,
                        'List Item' : 12,
                        'Map View' : 13,
                        'Multi-Tab' : 14,
                        'Number Stepper' : 15,
                        'On/Off Switch' : 16,
                        'Pager Indicator' : 17,
                        'Radio Button' : 18,
                        'Slider' : 19,
                        'Text' : 20,
                        'Text Button' : 21,
                        'Toolbar' : 22,
                        'Video' : 23,
                        'Web View' : 24,
                        'Icon' : 25}
    else :
        label_dictionary = {'Advertisement' : 1/num_classes,
                        'Background Image' : 2/num_classes,
                        'Bottom Navigation' : 3/num_classes,
                        'Button Bar' : 4/num_classes,
                        'Card' : 5/num_classes,
                        'Checkbox' : 6/num_classes,
                        'Drawer' : 7/num_classes,
                        'Date Picker' : 8/num_classes,
                        'Image' : 9/num_classes,
                        'Image Button' : 10/num_classes,
                        'Input' : 11/num_classes,
                        'List Item' : 12/num_classes,
                        'Map View' : 13/num_classes,
                        'Multi-Tab' : 14/num_classes,
                        'Number Stepper' : 15/num_classes,
                        'On/Off Switch' : 16/num_classes,
                        'Pager Indicator' : 17/num_classes,
                        'Radio Button' : 18/num_classes,
                        'Slider' : 19/num_classes,
                        'Text' : 20/num_classes,
                        'Text Button' : 21/num_classes,
                        'Toolbar' : 22/num_classes,
                        'Video' : 23/num_classes,
                        'Web View' : 24/num_classes,
                        'Icon' : 25/num_classes}
    return label_dictionary