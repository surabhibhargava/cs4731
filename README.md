# Classification and Image Generation from Sketches

## Image Generation

### Dependencies
- Tensorflow
- numpy 
- matplotlib
- opencv-python
- scikit-learn
- skimage

### Steps to set up training
1) Download the sketchy databse (https://drive.google.com/file/d/0B7ISyeE8QtDdTjE1MG9Gcy1kSkE/view) and place photos under ImageGeneration/datasets/sketchy/photos and sketches under ImageGeneration/datasets/sketchy/sketches. There should be folder for every class of images.

2) Run the script sketchy_data_prep.py to generate concatenated image sketch pairs needed for training. This computes distance transform of sketch and concatenates the image and sketch together.(The paths need to be modified to point to correct locations on the machine)

An image generated from this step should look like:</br>

![alt text](https://github.com/surabhibhargava/cs4731/blob/master/ImageGeneration/sketchy_concat/butterfly/n02274259_1147-1.jpg)

3) Run data_split.py to generate train, test and val splits with sketchy. (The paths need to be modified to point to correct locations on the machine)

4) To Train the model:
```
python main.py \
  --mode train \
  --dataset sketchy \
  --train_image_path ./sketchy_splits/butterfly/train \
  --test_image_path ./sketchy_splits/butterfly/val
```
  
5) To test the model:
```
python main.py \
  --mode test \ 
  --test_image_path ./sketchy_splits/butterfly/test \
  --checkpoint_name "path to ckpt"
```

Note: Make sure that the train, test, and val folders have images in them.

Sample results:
- For strawberry
![alt text](https://github.com/surabhibhargava/cs4731/blob/master/ImageGeneration/sample_results/strawberry.png "Strawberry")
- For Trees
![alt text](https://github.com/surabhibhargava/cs4731/blob/master/ImageGeneration/sample_results/trees.png "Tree")

References: </br>
- Sketchy Database: http://sketchy.eye.gatech.edu/
- Image to Image translation: https://arxiv.org/abs/1611.07004, https://github.com/prashnani/, https://phillipi.github.io/pix2pix/


