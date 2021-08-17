## Project Objectives:

<ul>
    <li>Implement the basic building blocks of ResNets in a deep neural network using Keras</li>
    <li>Put together these building blocks to implement and train a state-of-the-art neural network for image classification</li>
    <li>Implement a skip connection in your network</li>
</ul>
<br>
In ResNets, a "shortcut" or a "skip connection" allows the model to skip layers:

![skip_connection](./images/skip_connection_kiank.png)

<br>

By stacking these ResNet blocks on top of each other, you can form a very deep network.

<br>

ResNet blocks with the shortcut also makes it very easy for one of the blocks to learn an identity function. This means that we can stack on additional ResNet blocks with little risk of harming training set performance.

<br>

Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are the same or different.

#### The Identity Block:

The identity block is the standard block used in ResNets, and corresponds to the case where the input activation has the same dimension as the output activation. To speed up training, a BatchNorm step has been added.

![identity_skip_conn](./images/idblock3_kiank.png)

#### The Convolutional Block:

The ResNet "convolutional block" is the second block type. We can use this type of block when the input and output dimensions don't match up.

![conv_skip_conn](./images/convblock_kiank.png)

<br>

The CONV2D layer in the shortcut path is used to resize the input 𝑥 to a different dimension, so that the dimensions match up in the final addition needed to add the shortcut value back to the main path.

