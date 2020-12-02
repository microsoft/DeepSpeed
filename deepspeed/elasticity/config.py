class DeepSpeedElasticityConfig(object):
    def __init__(self, param_dict, world_size):
        super(DeepSpeedElasticityConfig, self).__init__()

        self.world_size = world_size

        self.micro_batches = []
        self.max_acceptable_batch_size = None
        self.min_gpus = None
        self.max_gpus = None
        self.prefer_larger_batch_size = True

        #batch size computed based on elasticity config
        self.computed_batch_size = None

        #micro batch size for this run computed based on
        #elasticity configs, and the world size
        self.computed_micro_batch = None

        #gradient accumulation steps for this run computed based on
        #elasticity configs, and the world size
        self.computed_gradient_accumulation_step = None

        self._initialize(param_dict)

    def _initialize(self, param_dict):
        self.micro_batches = []
        self.max_acceptable_batch_size = get_scalar_param()
        self.min_gpus = None
        self.max_gpus = None
        self.prefer_larger_batch_size = True

        #batch size computed based on elasticity config
        self.computed_batch_size = None

        #micro batch size for this run computed based on
        #elasticity configs, and the world size
        self.computed_micro_batch = None

        #gradient accumulation steps for this run computed based on
        #elasticity configs, and the world size
        self.computed_gradient_accumulation_step = None
