from .compress import get_module_name
from .helper import recursive_getattr
from deepspeed.utils import logger


class compression_scheduler():
    '''
    Used to schedule different compression methods
    '''
    def __init__(self, model, compression_config):
        self.model = model
        self.compression_config = compression_config
        self.make_init()
        self.training_steps = 0
        self.weight_quantization_enabled = False

        self.verbose = {
            "weight_quantization": False,
            "activation_quantization": False,
            "sparse_pruning": False,
            "head_pruning": False,
            "row_pruning": False,
            "channel_pruning": False,
        }

    def make_init(self):
        self.different_compression_methods = {}
        for method, method_content in self.compression_config:
            if "layer_reduction" in method:
                continue

            for group_name, method_parameters in method_content.different_groups.items():
                module_name_list = []
                for key_word in method_parameters.modules:
                    module_name, exist_module_name = get_module_name(group_name, self.model, key_word, exist_module_name, verbose=False)
                    module_name_list.extend(module_name)
                method_parameters.modules = module_name_list

    def check_weight_quantization(self):
        wq = self.compression_config.weight_quantization
        if not wq.enabled:
            return

        shared_parameters = wq.shared_parameters
        if self.training_steps >= shared_parameters.schedule_offset:
            for group_name, method_parameters in wq.different_groups.items():
                module_name_list = method_parameters.modules
                for module_name in module_name_list:
                    module = recursive_getattr(self.model, module_name)
                    module.weight_quantization_enabled = True

            if not self.verbose["weight_quantization"]:
                logger.info(
                    f'Weight quantization is enabled at step {self.training_steps}')
                self.weight_quantization_enabled = True
                self.verbose["weight_quantization"] = True

    def check_activation_quantization(self):
        # check activation quantization
        aq = self.compression_config.activation_quantization
        if not aq.enabled:
            return

        shared_parameters = aq.shared_parameters
        if self.training_steps >= shared_parameters.schedule_offset:
            for group_name, method_parameters in aq.different_groups.items():
                module_name_list = method_parameters.modules
                for module_name in module_name_list:
                    module = recursive_getattr(self.model, module_name)
                    module.activation_quantization_enabled = True
            if not self.verbose["activation_quantization"]:
                logger.info(
                    f'Activation quantization is enabled at step {self.training_steps}'
                )
                self.verbose["activation_quantization"] = True

    def check_sparse_pruning(self):
        # check sparse pruning
        sp = self.compression_config.sparse_pruning
        if not sp.enabled:
            return

        shared_parameters = sp.shared_parameters
        if self.training_steps >= shared_parameters.schedule_offset:
            for group_name, method_parameters in sp.different_groups.items():
                module_name_list = method_parameters.modules
                for module_name in module_name_list:
                    module = recursive_getattr(self.model, module_name)
                    module.sparse_pruning_enabled = True
            if not self.verbose["sparse_pruning"]:
                logger.info(
                    f'Sparse pruning is enabled at step {self.training_steps}')
                self.verbose["sparse_pruning"] = True

    def check_head_pruning(self):
        # check head pruning
        hp = self.compression_config.head_pruning
        if not hp.enabled:
            return

        shared_parameters = hp.shared_parameters
        if self.training_steps >= shared_parameters.schedule_offset:
            for group_name, method_parameters in hp.different_groups.items():
                module_name_list = method_parameters.modules
                for module_name in module_name_list:
                    module = recursive_getattr(self.model, module_name)
                    module.head_pruning_enabled = True
            if not self.verbose["head_pruning"]:
                logger.info(f'Head pruning is enabled at step {self.training_steps}')
                self.verbose["head_pruning"] = True

    def check_row_pruning(self):
        # check row pruning
        rp = self.compression_config.row_pruning
        if not rp.enabled:
            return

        shared_parameters = rp.shared_parameters
        if self.training_steps >= shared_parameters.schedule_offset:
            for group_name, method_parameters in rp.different_groups.items():
                module_name_list = method_parameters.modules
                for module_name in module_name_list:
                    module = recursive_getattr(self.model, module_name)
                    module.row_pruning_enabled = True
            if not self.verbose["row_pruning"]:
                logger.info(f'Row pruning is enabled at step {self.training_steps}')
                self.verbose["row_pruning"] = True

    def check_channel_pruning(self):
        # check channel pruning
        cp = self.compression_config.channel_pruning
        if not cp.enabled:
            return

        shared_parameters = cp.shared_parameters
        if self.training_steps >= shared_parameters.schedule_offset:
            for group_name, method_parameters in cp.different_groups.items():
                module_name_list = method_parameters.modules
                for module_name in module_name_list:
                    module = recursive_getattr(self.model, module_name)
                    module.channel_pruning_enabled = True
            if not self.verbose["channel_pruning"]:
                logger.info(
                    f'Channel pruning is enabled at step {self.training_steps}')
                self.verbose["channel_pruning"] = True

    def check_all_modules(self):
        # check all different compression methods we have
        self.check_weight_quantization()
        self.check_activation_quantization()
        self.check_sparse_pruning()
        self.check_head_pruning()
        self.check_row_pruning()
        self.check_channel_pruning()

    def step(self, step_zero_check=False):
        if not step_zero_check:
            self.training_steps += 1
        self.check_all_modules()
