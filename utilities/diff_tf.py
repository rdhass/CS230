import numpy as np
import tensorflow as tf

def ddx(inpt, channel, dx, scope='ddx', name=None):
    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    with tf.variable_scope(scope):
        ddx1D = tf.constant([-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.], dtype=tf.float32)
        ddx3D = tf.reshape(ddx1D, shape=(-1,1,1,1,1))

    strides = [1,1,1,1,1]
    var_pad = periodic_padding( var, ((3,3),(0,0),(0,0)) )
    output = tf.nn.conv3d(var_pad, ddx3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    output = tf.scalar_mul(1./dx, output)
    
    return output


def ddy(inpt, channel, dy, scope='ddy', name=None):
    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    with tf.variable_scope(scope):
        ddy1D = tf.constant([-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.], dtype=tf.float32)
        ddy3D = tf.reshape(ddy1D, shape=(1,-1,1,1,1))

    strides = [1,1,1,1,1]
    var_pad = periodic_padding( var, ((0,0),(3,3),(0,0)) )
    output = tf.nn.conv3d(var_pad, ddy3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    output = tf.scalar_mul(1./dy, output)
    
    return output


def ddz(inpt, channel, dz, scope='ddz', name=None):
    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    with tf.variable_scope(scope):
        ddz1D = tf.constant([-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.], dtype=tf.float32)
        ddz3D = tf.reshape(ddz1D, shape=(1,1,-1,1,1))

    strides = [1,1,1,1,1]
    var_pad = periodic_padding( var, ((0,0),(0,0),(3,3)) )
    output = tf.nn.conv3d(var_pad, ddz3D, strides, padding = 'VALID',
                          data_format = 'NDHWC', name=name)
    output = tf.scalar_mul(1./dz, output)
    
    return output
