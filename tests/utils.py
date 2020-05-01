import tensorflow as tf


def transfer(x, func):
    with tf.GradientTape() as g:
        x_tensor = tf.constant(x)
        g.watch(x_tensor)
        y_tensor = func(x_tensor)
        dx_tensor = g.gradient(y_tensor, x_tensor)

    return y_tensor.numpy(), dx_tensor.numpy()
