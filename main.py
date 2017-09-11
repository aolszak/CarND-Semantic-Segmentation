import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# Functions
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    vgg_input_tensor_name = graph.get_tensor_by_name('image_input:0')
    vgg_keep_prob_tensor_name = graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out_tensor_name = graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out_tensor_name = graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out_tensor_name = graph.get_tensor_by_name('layer7_out:0')

    return vgg_input_tensor_name, vgg_keep_prob_tensor_name, vgg_layer3_out_tensor_name, vgg_layer4_out_tensor_name, vgg_layer7_out_tensor_name
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # Convolution of every layer from function parameters to have always the same shape in FCN
    vgg_layer3_logits = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, name='vgg_layer3_logits')
    vgg_layer4_logits = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, name='vgg_layer4_logits')
    vgg_layer7_logits = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, name='vgg_layer7_logits')

    # Transposed convolution layer from vgg_layer7_logits
    fcn_decoder_layer1 = tf.layers.conv2d_transpose(vgg_layer7_logits, num_classes, kernel_size=4, strides=(2, 2), padding='same', name='fcn_decoder_layer1')

    # Skip layer from fcn_decoder_layer1 and vgg_layer4_logits
    fcn_decoder_layer2 = tf.add(fcn_decoder_layer1, vgg_layer4_logits, name='fcn_decoder_layer2')

    # Transposed convolution layer from fcn_decoder_layer2
    fcn_decoder_layer3 = tf.layers.conv2d_transpose(fcn_decoder_layer2, num_classes, kernel_size=4, strides=(2, 2), padding='same', name='fcn_decoder_layer3')

    # Skip layer from fcn_decoder_layer3 and vgg_layer4_logits
    fcn_decoder_layer4 = tf.add(fcn_decoder_layer3, vgg_layer3_logits, name='fcn_decoder_layer4')

    # Last transposed convolution layer from fcn_decoder_layer4
    fcn_decoder_layer5 = tf.layers.conv2d_transpose(fcn_decoder_layer4, num_classes, kernel_size=16, strides=(8, 8), padding='same', name='fcn_decoder_layer5')

    return fcn_decoder_layer5
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # Reshape the last layer and labels
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Loss function and Adam optimizer
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    #
    # Start time

    total_start_time = time.clock()       

    #
    # Training

    sess.run(tf.global_variables_initializer())

    for i in range(epochs):

        # Epoch start information
        epoch_num = i+1
        print("Epoch: {}".format(epoch_num))

        # Epoch variables
        training_loss = 0
        training_samples = 0
        start_time = time.clock()

        # Train with batches
        for X, y in get_batches_fn(batch_size):
            training_samples += len(X)
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={input_image: X, correct_label: y, keep_prob: 0.8})
            training_loss += loss

        # Training loss
        training_loss /= training_samples

        # Epoch summary log message
        end_time = time.clock()
        training_time = end_time - start_time
        print("Epoch: {}, time: {} seconds, training loss: {}".format(epoch_num, training_time, training_loss))

    # Total time log message
    total_end_time = time.clock()
    total_time = total_end_time - total_start_time
    print("Total time: {} seconds".format(total_time))

    pass
tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs = 15
    batch_size = 1
    lr = 0.0001
    learning_rate = tf.constant(lr)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        vgg_input, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess=sess, epochs=epochs, batch_size=batch_size, get_batches_fn=get_batches_fn, train_op=train_op, cross_entropy_loss=cross_entropy_loss, input_image=vgg_input, correct_label=correct_label, keep_prob=keep_prob, learning_rate=lr)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, vgg_input)        

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':

    # GPU check
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU. prepare yourself for long training ;-)')

    # Test
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn)
    tests.test_for_kitti_dataset('./data')

    run()