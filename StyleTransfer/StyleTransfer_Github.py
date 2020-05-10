import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import PIL.Image
import functools
import os

# Training iterations
Train_iterations = 5

# Style and Content weights
style_weight=.01
content_weight=10000

# Choose Layers for conserving content, and style
content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Converts Content Image Variable, to image
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    tensor = tensor[0] # (422, 512, 3) Choose the first IMAGE
    return PIL.Image.fromarray(tensor)

# Load Files, local directory path to your image
content_path = r'CONTENTIMAGEPATH.jpg'
style_path = r'STYLEIMAGEPATH.jpg'

# Create Individual File name
content_base = os.path.basename(content_path)
style_base = os.path.basename(style_path)
output_file_name = content_base.split('.')[0] + style_base.split('.')[0]


# Rewrite last try new scale
def load_img(path_to_img):
    max_dim = 512 # No reason, just the scale they decided on
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32) #removes the last dimension, converts that shape into a float for manipulation
    long_dim = max(shape) # get max argument val
    scale = max_dim / long_dim #Scale used to keep the long dim, and reduce the shorter one

    new_shape = tf.cast(shape * scale, tf.int32) # apply scaling and round to an int

    img = tf.image.resize(img, new_shape) # tensor flow method for resizing and maintaining the image contents, resizes firs two dimensions to the ones provided bilinear interpollation makes range [0,1]
    img = img[tf.newaxis, :] # just adds an axis for the batch in, and keeps the rest of the image
    print(img.shape) #(1, 422, 512, 3)
    return img


# Get Images
content_image = load_img(content_path)
style_image = load_img(style_path)


def vgg_layers(layer_names):
    """ Returns a model that has the outputs of the layers that are provided for the vgg network """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False  # lock the model

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    m, n_H, n_W, n_C = list(input_tensor.shape)
    a = tf.transpose(tf.reshape(input_tensor, [n_H * n_W, n_C]))
    GA = tf.matmul(a, tf.transpose(a))
    return GA


def initializemodel():
    vgg = vgg_layers(style_layers + content_layers)
    vgg.trainable = False
    return vgg

vgg = initializemodel()


def StyleTransferModel(image_input):
    image_input = image_input * 255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(image_input)
    outputs = vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:num_style_layers], outputs[num_style_layers:])

    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

    content_dict = {content_name: value for content_name, value in zip(content_layers, content_outputs)}
    style_dict = {style_name: value for style_name, value in zip(style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

style_targets = StyleTransferModel(style_image)['style']
content_targets = StyleTransferModel(content_image)['content']
image = tf.Variable(content_image)


def clip(image): # ensure values are between 0 and 1 after variable update
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


def total_loss(outputs):
    style_out = outputs['style']
    content_out = outputs['content']

    style_loss = 0
    for layer in style_out.keys():
        style_loss += tf.reduce_mean((style_out[layer] - style_targets[layer])**2)

    style_loss = style_loss/num_style_layers

    content_loss = 0
    for layer in content_out.keys():
        content_loss += tf.reduce_mean((content_out[layer] - content_targets[layer]) ** 2)

    content_loss = content_loss / num_content_layers

    loss = style_loss*style_weight + content_loss*content_weight
    return loss


@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = StyleTransferModel(image)
    loss = total_loss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip(image))


#Train and update the image
def train_image(Train_iterations,image):
    for m in range(Train_iterations):
        train_step(image)
        if m % 50 == 0:
            print("Train step: {}".format(m))

    file_name = output_file_name + '.png'
    tensor_to_image(image).save(file_name)

train_image(Train_iterations, image)

