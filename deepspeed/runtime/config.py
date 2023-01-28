from pydantic import Field, validator, root_validator
from deepspeed import comm as dist
from deepspeed.config_utils import DeepSpeedConfigModel, DtypeEnum
from deepspeed.runtime.zero.config import DeepSpeedZeroConfig


class DeepSpeedFP16Config(DeepSpeedConfigModel):
    enabled: bool = False
    master_weights_and_grads: bool = Field(False, alias="fp16_master_weights_and_grads")
    auto_cast: bool = False
    loss_scale: float = Field(0, ge=0)
    initial_scale_power: float = Field(16, ge=0)
    loss_scale_window: float = Field(1000, ge=0)
    hysteresis: float = Field(2, ge=0)
    min_loss_scale: float = Field(1, gt=0)


class DeepSpeedBF16Config(DeepSpeedConfigModel):
    enabled: bool = False


class DeepSpeedAMPConfig(DeepSpeedConfigModel):
    enabled: bool = False

    @property
    def params(self):
        amp_params = self.dict()
        amp_params.pop("enabled")
        return amp_params

    class Config:
        extra = "allow"  # This config can be populated by any kwargs that AMP supports


class DeepSpeedOptimizerConfig(DeepSpeedConfigModel):
    type: str = None
    params: Dict[str, Any] = None
    legacy_fusion: bool = False


class DeepSpeedSchedulerConfig(DeepSpeedConfigModel):
    type: str = None
    params: Dict[str, Any] = None


class DeepSpeedEigenvalueConfig(DeepSpeedConfigModel):
    enabled: bool = False
    verbose: bool = False
    max_iter: int = Field(100, ge=0)
    tol: float = Field(1e-2, ge=0)
    stability: float = Field(1e-6, ge=0)
    gas_boundary_resolution: int = Field(1, ge=0)
    layer_name: str = "bert.encoder.layer"
    layer_num: int = Field(0, ge=0)


class DeepSpeedPipelineConfig(DeepSpeedConfigModel):
    stages: str = "auto"  # TODO: convert to Enum class
    partition: str = "best"  # TODO: convert to Enum class
    seed_layers: bool = False
    activation_checkpoint_interval: int = Field(0, ge=0)


class DeepSpeedPLDConfig(DeepSpeedConfigModel):
    enabled: bool = False
    theta: float = Field(1.0, ge=0)
    gamma: float = Field(0.001, ge=0)

    @property
    def params(self):
        if not self.enabled:  # TODO: Check if this can be removed
            return False
        pld_params = self.dict()
        pld_params.pop("enabled")
        return pld_params


class CheckpointValidationEnum(str, Enum):
    IGNORE: "IGNORE"
    WARN: "WARN"
    FAIL: "FAIL"


class ParallelWriteConfig(DeepSpeedConfigModel):
    pipeline_stage: bool = False


class DeepSpeedCheckpointConfig(DeepSpeedConfigModel):
    tag_validation: CheckpointValidationEnum = "Warn"
    load_universal: bool = False
    use_node_local_storage: bool = False
    parallel_write: ParallelWriteConfig = {}

    @validator("tag_validation", pre=True)
    def upper_case_str(cls, field_value, values):
        return field_values.upper()


class DeepSpeedDataTypesConfig(DeepSpeedConfigModel):
    grad_accum_dtype: DtypeEnum = None


class DeepSpeedConfig(DeepSpeedConfigModel):
    mpu: object = None
    global_rank: int = Field(0, ge=0)
    world_size: int = Field(1, ge=1)

    train_batch_size: int = Field(None, ge=1)
    train_micro_batch_size_per_gpu: int = Field(None, ge=1)
    gradient_accumulation_steps: int = Field(None, ge=1)
    steps_per_print: int = Field(10, ge=1)
    dump_state: bool = False

    disable_allgather: bool = False
    communication_data_type: DtypeEnum = None
    prescale_gradients: bool = False
    gradient_predivide_factor: float = Field(1.0, ge=0)
    sparse_gradients: bool = False

    zero_optimization: DeepSpeedZeroConfig = {}
    activation_checkpointing: DeepSpeedActivationCheckpointingConfig = {}
    comms_logger: DeepSpeedCommsConfig = {}
    monitor_config: DeepSpeedMonitorConfig = Field(
        {}
    )  # TODO csv_monitor, wandb, tensorboard are values that need to be placed into this config
    compression_training: DeepSpeedCompressionConfig = {}
    flops_profiler: DeepSpeedFlopsProfilerConfig = {}
    nebula: DeepSpeedNebulaConfig = {}
    fp16: DeepSpeedFP16Config = {}
    bf16: DeepSpeedBF16Config = Field(
        {},
        alias="bfloat16")  # Alias for backward compatibility
    optimizer: DeepSpeedOptimizerConfig = {}
    scheduler: DeepSpeedSchedulerConfig = {}
    autotuning: DeepSpeedAutotuningConfig = {}
    amp: DeepSpeedAMPConfig = {}
    eigenvalue: DeepSpeedEigenvalueConfig = {}
    pipeline: DeepSpeedPipelineConfig = {}
    progressive_layer_drop: DeepSpeedPLDConfig = {}
    curriculum_learning: DeepSpeedCurriculumLearningConfig = {}
    data_efficiency: DeepSpeedDataEfficiencyConfig = {}
    checkpoint: DeepSpeedCheckpointConfig = {}
    data_types: DeepSpeedDataTypeConfig = {}
    aio: DeepSpeedAIOConfig = {}
    elasticity: DeepSpeedElasticityConfig = {}

    gradient_clipping: float = Field(0.0, ge=0)
    zero_allow_untested_optimizer: bool = False
    memory_breakdown: bool = False
    sparse_attention: bool = False
    wall_clock_breakdown: bool = False
    dataloader_drop_last: bool = False

    # Theses are here for backward compatibility with any downstream
    # applications that use the ds_config directly, but should be removed
    # before/at v1.0 release
    @property
    def zero_config(self):
        return self.zero_optimization

    @property
    def activation_checkpointing_config(self):
        return self.activation_checkpointing.dict()

    @property
    def comms_config(self):
        return self.comms_logger.dict()

    @property
    def compression_config(self):
        return self.compression_training.dict()

    @property
    def flops_profiler_config(self):
        return self.flops_profiler.dict()

    @property
    def nebula_config(self):
        return self.nebula.dict()

    @property
    def autotuning_config(self):
        return self.autotuning.dict()

    @property
    def aio_config(self):
        return self.aio.dict()

    @property
    def zero_enabled(self):
        return bool(self.zero_config.stage > 0)

    @property
    def fp16_enabled(self):
        return self.fp16_config.enabled

    @property
    def fp16_auto_cast(self):
        return self.fp16_config.autocast

    @property
    def fp16_master_weights_and_gradients(self):
        return self.fp16_config.master_weights_and_grads

    @property
    def loss_scale(self):
        return self.fp16_config.loss_scale

    @property
    def initial_dynamic_scale(self):
        if self.bf16_enabled:
            return 0
        return 2**self.fp16_config.initial_scale_power

    @property
    def dynamic_loss_scale_args(self):
        if not self.fp16_enabled:
            return None
        loss_scale_args = {
            INITIAL_LOSS_SCALE: 2**self.fp16_config.initial_scale_power,
            SCALE_WINDOW: self.fp16_config.loss_scale_window,
            DELAYED_SHIFT: self.fp16_config.hysteresis,
            MIN_LOSS_SCALE: self.fp16_config.min_loss_scale,
        }
        return loss_scale_args

    @property
    def optimizer_name(self):
        opt_type = self.optimizer_config.type
        if opt_type is None:
            return opt_type
        elif opt_type.lower() in DEEPSPEED_OPTIMIZER:
            return opt_type.lower()
        else:
            return opt_type

    @property
    def optimizer_params(self):
        return self.optimizer_config.params

    @property
    def optimizer_legacy_fusion(self):
        return self.optimizer_config.legacy_fusion

    @property
    def scheduler_name(self):
        return self.scheduler_config.type

    @property
    def scheduler_params(self):
        return self.scheduler_config.params

    @property
    def wall_clock_breakdown(self):
        return self.wall_clock_breakdown | self.flops_profiler_config.enabled

    @property
    def bfloat16_enabled(self):
        return self.bf16.enabled

    @property
    def amp_enabled(self):
        return self.amp.enabled

    @property
    def amp_params(self):
        return self.amp.params

    @property
    def eigenvalue_enabled(self):
        return self.eigenvalue.enabled

    @property
    def eigenvalue_verbose(self):
        return self.eigenvalue.verbose

    @property
    def eigenvalue_max_iter(self):
        return self.eigenvalue.max_iter

    @property
    def eigenvalue_tol(self):
        return self.eigenvalue.tol

    @property
    def eigenvalue_stability(self):
        return self.eigenvalue.stability

    @property
    def eigenvalue_gas_boundary_resolution(self):
        return self.eigenvalue.gas_boundary_resolution

    @property
    def eigenvalue_layer_name(self):
        return self.eigenvalue.layer_name

    @property
    def eigenvalue_layer_num(self):
        return self.eigenvalue.layer_num

    @property
    def pld_enabled(self):
        return self.progressive_layer_drop.enabled

    @property
    def pld_params(self):
        return self.progressive_layer_drop.params

    @property
    def curriculum_enabled_legacy(self):
        return self.curriculum_learning.enabled

    @property
    def curriculum_params_legacy(self):
        return self.curriculum_learning.params

    @property
    def data_efficiency_enabled(self):
        return self.data_efficiency.enabled

    @property
    def data_efficiency_config(self):
        return self.data_efficiency.config

    @property
    def checkpoint_tag_validation_enabled(self):
        return self.checkpoint.tag_validation != CheckpointValidationEnum.IGNORE

    @property
    def checkpoint_tag_validation_enabled(self):
        return self.checkpoint.tag_validation == CheckpointValidationEnum.FAIL

    @property
    def load_universal_checkpoint(self):
        return self.checkpoint.load_universal

    @property
    def use_node_local_storage(self):
        return self.checkpoint.use_node_local_storage

    @property
    def grad_accum_dtype(self):
        return self.data_types.grad_accum_dtype

    @property
    def checkpoint_parallel_write_pipeline(self):
        return self.checkpoint.parallel_write.pipeline_stage

    @property
    def elasticity_enabled(self):
        return self.elasticity.enabled

    # Validation functions
    @validator("global_rank")
    def get_global_rank(cls, field_value, values):
        try:
            field_value = dist.get_rank()
        except:
            pass
        return field_value

    @validator("world_size")
    def get_world_size(cls, field_value, values):
        try:
            if values.get("mpu") != None:
                field_value = mpu.get_data_parallel_world_size()
            else:
                field_value = dist.get_world_size()
        except:
            pass
        return field_value

    @root_validator
    def _exclusive_fp16_bf16(cls, values):
        assert not (
            values.get("bf16").enabled and values.get("fp16").enabled
        ), "bf16 and fp16 modes cannot be simultaneously enabled"
        return values

    # TODO: Make root_validator
    def _batch_assertion(self, values):
        train_batch = values.get("train_batch_size")
        micro_batch = values.get("train_micro_batch_size_per_gpu")
        grad_acc = values.get("gradient_accumulation_steps")
        world_size = values.get("world_size")

        assert (
            train_batch > 0
        ), f"Train batch size: {train_batch} has to be greater than 0"

        assert (
            micro_batch > 0
        ), f"Micro batch size per gpu: {micro_batch} has to be greater than 0"

        assert (
            grad_acc > 0
        ), f"Gradient accumulation steps: {grad_acc} has to be greater than 0"

        assert train_batch == micro_batch * grad_acc * world_size, (
            f"Check batch related parameters. train_batch_size is not equal "
            "to micro_batch_per_gpu * gradient_acc_step * world_size "
            f"{train_batch} != {micro_batch} * {grad_acc} * {world_size}"
        )

    @root_validator
    def _set_batch_related_parameters(cls, values):
        train_batch = values.get("train_batch_size")
        micro_batch = values.get("train_micro_batch_size_per_gpu")
        grad_acc = values.get("gradient_accumulation_steps")
        world_size = values.get("world_size")

        # all values are provided nothing needs to be set
        if train_batch is not None and micro_batch is not None and grad_acc is not None:
            pass

        # global_accumulation_steps needs to be set
        elif train_batch is not None and micro_batch is not None:
            grad_acc = train_batch // micro_batch
            grad_acc //= world_size
            values["gradient_accumulation_steps"] = grad_acc

        # micro_batch_per_gpu needs to be set
        elif train_batch is not None and grad_acc is not None:
            micro_batch = train_batch // world_size
            micro_batch //= grad_acc
            values["train_micro_batch_size_per_gpu"] = micro_batch

        # train_batch_size needs to be set
        elif micro_batch is not None and grad_acc is not None:
            train_batch_size = micro_batch * grad_acc
            train_batch_size *= world_size
            values["train_batch_size"] = train_batch_size

        # gradient_accumulation_steps and micro_batch_per_gpus is set
        elif train_batch is not None:
            values["gradient_accumulation_steps"] = 1
            values["train_micro_batch_size_per_gpu"] = train_batch // world_size

        # train_batch_size and gradient_accumulation_step is set
        elif micro_batch is not None:
            values["train_batch_size"] = micro_batch * world_size
            values["gradient_accumulation_steps"] = 1

        # either none of the three parameters are provided or just gradient_accumulation_step is provided
        else:
            assert (
                False
            ), "Either train_batch_size or train_micro_batch_size_per_gpu needs to be provided"

        # TODO: Make this another root validator, figure out how to order root_validator
        self._batch_assertion(values)

        return values
