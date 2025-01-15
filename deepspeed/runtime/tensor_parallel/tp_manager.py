import torch
from .config import DeepSpeedTPTrainingConfig, DeepSpeedTPConfig
from deepspeed.utils import groups




class TpTrainingManager():
    def __init__(self, model, tp_size, dtype):
        self.module = model
        self.config = self._initialize_config(dtype)
        
        from deepspeed.module_inject.auto_tp import AutoTP  

        # Parse model configuration
        parser_dict = AutoTP.tp_parser(model)
        print("AutoTP: ", parser_dict)
        
        # Initialize TP configuration and model
        self._initialize_tp_config(tp_size)
        self._get_model_config_generate()
        
        # Apply injection policies
        self._apply_policies(parser_dict)
        
    def _initialize_config(self, dtype):
        """Initialize and return the DeepSpeed TP training configuration."""
        config = DeepSpeedTPTrainingConfig()
        config.dtype = dtype
        return config
    
    def _apply_policies(self, parser_dict):
        """Apply injection policies to the parsed modules."""
        for client_module, injection_policy in parser_dict:
            self.config.injection_policy_tuple = injection_policy
            self._apply_injection_policy(self.config, client_module)
    
    def _apply_injection_policy(self, config, client_module=None):
        from deepspeed.module_inject import replace_transformer_layer

        """Apply the given injection policy to a client module."""
        # replace_transformer_layer(client_module, self.module, None, self.config, self.model_config)
        if isinstance(self.module, torch.nn.Module):
            replace_transformer_layer(client_module, self.module, None, self.config, self.model_config)
        
    def _initialize_tp_config(self, tp_size):
        """Perform TP configuration initialization."""
        self.tp_config=DeepSpeedTPConfig()
        self.tp_config.tp_size =tp_size
        if tp_size <= 1:
            self.tp_config.enabled = False
        groups._init_tp_mesh_device(tp_size)
        self.tp_config.tp_group = groups.get_tensor_model_parallel_group()
        self.config.tensor_parallel = self.tp_config
        
        
    def _get_model_config_generate(self):
        """Generate and apply HF model  configuration."""
        self.model_config = getattr(self.module, 'config', None)