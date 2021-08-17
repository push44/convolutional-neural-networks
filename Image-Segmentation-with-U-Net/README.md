## Project Objectives:

<ul>
    <li>Build your own U-Net</li>
    <li>Implement semantic image segmentation on the CARLA self-driving car dataset</li>
    <li>Apply sparse categorical crossentropy for pixelwise prediction</li>
</ul>

We'll be building your own U-Net, a type of CNN designed for quick, precise image segmentation, and using it to predict a label for every single pixel in an image - in this case, an image from a self-driving car dataset.

![carseg.png](./images/carseg.png)

## U-Net Model:

![unet](./images/unet.png)

## Loss Function:

In semantic segmentation, we need as many masks as we have object classes. In the dataset we're using, each pixel in every mask has been assigned a single integer probability that it belongs to a certain class, from 0 to num_classes-1. The correct class is the layer with the higher probability. 

<br>

This is different from categorical crossentropy, where the labels should be one-hot encoded (just 0s and 1s). Here, we'll use sparse categorical crossentropy as our loss function, to perform pixel-wise multiclass prediction. Sparse categorical crossentropy is more efficient than other loss functions when we're dealing with lots of classes.

## Dataset Handling:

Note that in image segmentation our ground truth label is the true mask. True mask is what our trained model output is aiming to et as close as possible.

