# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 09:18:01 2018

@author: shen1994
"""

import os
import tensorflow as tf
from keras import backend as K
from model import DS2_model
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework import graph_io

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    input_shape=(1960, 161)
    output_shape=(219, )
    
    # define model
    base_model = DS2_model(input_shape=input_shape, output_shape=output_shape) 
    base_model.load_weights('model/weights.169.hdf5', by_name=True)

    # boxes = tf.placeholder(dtype='float32', shape=(None, 4))
    # scores = tf.placeholder(dtype='float32', shape=(None,))
    # nms = tf.image.non_max_suppression(boxes, scores, 100, iou_threshold=0.45)
    print('input name is: ', base_model.input.name)
    print('output name is: ', base_model.output.name)
    
    K.set_learning_phase(0)
    frozen_graph = freeze_session(K.get_session(), output_names=[base_model.output.op.name])
    graph_io.write_graph(frozen_graph, "model/", "pico_DeepDS2_model.pb", as_text=False)
   
    
    
    
    
    
    
    
    
    
    
    
    