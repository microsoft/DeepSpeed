class TensorLogger():
    """Initialize a tensor logger
    """
    def __init__(self):
        self.tensor_dict = {}


    def add_tensor(self, tensor, tensor_id, group_id):
        """Log a tensor to the tensor dictionary

            Arguments:
                tensor: Torch Tensor for logging

                tensor_id: Unique Tensor ID corresponding to the tensor being logged

                group_id: Group ID (i.e. key) corresponding to the group of tensors being logged. This key can be a string,
                          int, etc.
        """
        if group_id in self.tensor_dict:
            self.tensor_dict[group_id][tensor_id] = tensor
        else:
            self.tensor_dict.update({group_id:{tensor_id:tensor}})


    def remove_tensor(self, tensor, tensor_id, group_id):
        """Remove a tensor from the tensor dictionary

            Arguments:
                tensor: Torch Tensor for logging

                tensor_id: Unique Tensor ID corresponding to the tensor being logged

                group_id: Group ID (i.e. key) corresponding to the group of tensors being logged. This key can be a string,
                          int, etc.
        """
