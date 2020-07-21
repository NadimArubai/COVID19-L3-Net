# from . import semseg_cost
import torch
import os
import tqdm 
from . import semseg
from . import semseg_for_onnx
from . import semseg_for_onnx_export
import torch


def get_model(model_dict, exp_dict=None, train_set=None):
    if model_dict['name'] in ["semseg"]:
        model =  semseg.SemSeg(exp_dict, train_set)

        # load pretrained
        if 'pretrained' in model_dict:
            model.load_state_dict(torch.load(model_dict['pretrained']))
 
    return model

def get_model_for_onnx(model_dict, exp_dict=None, train_set=None):
    if model_dict['name'] in ["semseg"]:
        model =  semseg_for_onnx.SemSeg(exp_dict, train_set)

        # load pretrained
        if 'pretrained' in model_dict:
            model.load_state_dict(torch.load(model_dict['pretrained']))
 
    return model


def get_model_for_onnx_export(model_dict, exp_dict=None, train_set=None):
    if model_dict['name'] in ["semseg"]:
        model =  semseg_for_onnx_export.SemSeg(exp_dict, train_set)

        # load pretrained
        if 'pretrained' in model_dict:
            model.load_state_dict(torch.load(model_dict['pretrained']))
 
    return model
