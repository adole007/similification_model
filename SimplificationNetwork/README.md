# Stroke Simplification Network

## Setting up the project

There is a Pipfile included that can be used with Pipenv to setup the python environment.
Details of setting up and using Pipenv can be found here: https://github.com/pypa/pipenv

## Generating Data

First step is to prepare the data set for training. The gen_simplify.py script does
this for you - you need to configure the script to point at the input data and the output
location, then run the script.

1. Edit config.json
* Create a base directory and set base_dir
* Update gen->root_dir to point at the training data (i.e. the folder that contains image and path files)
2. Run the gen_simplify.py script
3. Check that the root dir contains train, validate and test folders with img and seg folders are populated

## Training

Update config.json and run the simplify.py script. If you are using tensorboard, create a log directory and
set base_log_dir to point at it.

## Notes

### GPU Constraints

Depending on the GPU available, you might need to generate smaller images for training and update the target size
in config.json.

### Network Definition

The network is constructed in model_simplify.py - edit this to change the network 
architecture.

### Data Augmentation

The training data is augmented on the fly by a set of methods in generator.py. The probability
any particular augmentation is applied is defined in this file. Current augmentations:
1. Noise - adds noise to the image randomly from a set of different methods
2. Budge - creates a copy of the image and merges with the original with some small offset
3. Blur - adds a Gaussian blur to the training image
4. Contrast - stretches or compresses the image contrast randomly within a range

