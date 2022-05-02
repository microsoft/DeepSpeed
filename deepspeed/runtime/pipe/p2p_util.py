from enum import Enum
import torch
import typing

class Type(Enum):
    TUPLE = 1
    LIST = 2
    DICT = 3
    TENSOR = 4

class Data_Type(Enum):
    FLOAT = 1
    FLOAT16 = 2
    FLOAT64 = 3
    DOUBLE = 4
    LONG = 5
    BFLOAT16 = 6
    HALF = 7
    UINT8 = 8
    INT8 = 9
    INT16 = 10
    SHORT = 11
    INT = 12
    INT64 = 13
    BOOL = 14

def encode_type(object : typing.Any):
    if isinstance(object, tuple):
        return Type.TUPLE
    elif isinstance(object, list):
        return Type.LIST
    elif isinstance(object, dict):
        return Type.DICT
    elif isinstance(object, torch.Tensor):
        return Type.TENSOR
    else:
        raise Exception("Type is not supported for encoding: ", type(object))    

def encode_element_type(tensor : torch.Tensor) -> int:
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float:
        return Data_Type.FLOAT.value
    elif tensor.dtype == torch.float16:
        return Data_Type.FLOAT16.value
    elif tensor.dtype == torch.float64:
        return Data_Type.FLOAT64.value
    elif tensor.dtype == torch.double:
        return Data_Type.DOUBLE.value
    elif tensor.dtype == torch.long:
        return Data_Type.LONG.value
    elif tensor.dtype == torch.bfloat16:
        return Data_Type.BFLOAT16.value
    elif tensor.dtype == torch.half:
        return Data_Type.HALF.value
    elif tensor.dtype == torch.uint8:
        return Data_Type.UINT8.value
    elif tensor.dtype == torch.int8:
        return Data_Type.INT8.value
    elif tensor.dtype == torch.int16:
        return Data_Type.INT16.value
    elif tensor.dtype == torch.short:
        return Data_Type.SHORT.value
    elif tensor.dtype == torch.int or tensor.dtype == torch.int32:
        return Data_Type.INT.value
    elif tensor.dtype == torch.int64:
        return Data_Type.INT64.value
    elif tensor.dtype == torch.bool:
        return Data_Type.BOOL.value
    else:
        raise Exception("Tensor element type is not supported: ", tensor.dtype)


def decode_element_type(type : int):
    if type == 1: 
        return torch.float
    elif type == 2:
        return torch.float16
    elif type == 3:
        return torch.float64
    elif type == 4:
        return torch.double
    elif type == 5:
        return torch.long
    elif type == 6:
        return torch.bfloat16
    elif type == 7:
        return torch.half
    elif type == 8:
        return torch.uint8
    elif type == 9:
        return torch.int8
    elif type == 10:
        return torch.int16
    elif type == 11:
        return torch.short
    elif type == 12:
        return torch.int
    elif type == 13:
        return torch.int64
    elif type == 14:
        return torch.bool
    else:
        raise Exception("Invalid type id: ", type)

