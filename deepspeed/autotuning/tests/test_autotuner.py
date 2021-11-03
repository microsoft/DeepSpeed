import argparse
from deepspeed.autotuning.autotuner import DSAutotuner
from deepspeed.launcher.runner import DLTS_HOSTFILE, TORCH_DISTRIBUTED_DEFAULT_PORT

TEST_USER_SCRIPT = "/home/chengli1/projects/transformers/examples/pytorch/text-classification/run_glue.py"
TEST_USER_ARGS = "--autotune --deepspeed ds_config.json \
  --model_name_or_path microsoft/deberta-v2-xxlarge \
  --task_name ${TASK_NAME} \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size ${batch_size} \
  --learning_rate 13e-6 \
  --num_train_epochs 3\
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --logging_dir ${output_dir} \
  --save_steps 0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e',
                        '--exps_dir',
                        type=str,
                        default="",
                        help="output directory for the generated experiments")
    parser.add_argument('-r',
                        '--results_dir',
                        type=str,
                        default="",
                        help="output directory for the experiments results")
    parser.add_argument("-H",
                        "--hostfile",
                        type=str,
                        default=DLTS_HOSTFILE,
                        help="Hostfile path (in MPI style) that defines the "
                        "resource pool available to the job (e.g., "
                        "worker-0 slots=4)")
    parser.add_argument(
        '-log',
        '--loglevel',
        default='debug',
        help='Provide logging level. Example --loglevel debug, default=warning')

    parser.add_argument("--master_port",
                        default=TORCH_DISTRIBUTED_DEFAULT_PORT,
                        type=int,
                        help="(optional) Port used by PyTorch distributed for "
                        "communication during training.")

    parser.add_argument("--user_script",
                        default=TEST_USER_SCRIPT,
                        type=str,
                        help="user python script")

    parser.add_argument("--user_args", type=str, help="user args")
    args = parser.parse_args()

    autotuner = DSAutotuner(args, None)
    autotuner.logger.setLevel(args.loglevel.upper())
    autotuner.tune()
