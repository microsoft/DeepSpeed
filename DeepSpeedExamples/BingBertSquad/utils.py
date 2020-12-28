import os
import sys
import logging
import argparse
from tensorboardX import SummaryWriter

SUMMARY_WRITER_DIR_NAME = 'runs'


def get_argument_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model checkpoints and predictions will be written."
    )

    # Other parameters
    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json"
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help=
        "When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help=
        "The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--predict_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for predictions.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help=
        "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
        "of training.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help=
        "The total number of n-best predictions to generate in the nbest_predictions.json "
        "output file.")
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help=
        "The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.")
    parser.add_argument(
        "--verbose_logging",
        action='store_true',
        help=
        "If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help=
        "Whether to lower case the input text. True for uncased models, False for cased models."
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--fp16',
        action='store_true',
        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument(
        '--wall_clock_breakdown',
        action='store_true',
        default=False,
        help=
        "Whether to display the breakdown of the wall-clock time for foraward, backward and step"
    )
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help=
        "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--model_file",
                        type=str,
                        default="0",
                        help="Path to the Pretrained BERT Encoder File.")

    parser.add_argument("--max_grad_norm",
                        default=1.,
                        type=float,
                        help="Gradient clipping for FusedAdam.")
    parser.add_argument('--job_name',
                        type=str,
                        default=None,
                        help='Output path for Tensorboard event files.')

    parser.add_argument(
        '--preln',
        action='store_true',
        default=False,
        help=
        "Whether to display the breakdown of the wall-clock time for foraward, backward and step"
    )

    parser.add_argument(
        '--loss_plot_alpha',
        type=float,
        default=0.2,
        help='Alpha factor for plotting moving average of loss.')

    parser.add_argument(
        '--max_steps',
        type=int,
        default=sys.maxsize,
        help=
        'Maximum number of training steps of effective batch size to complete.'
    )

    parser.add_argument(
        '--max_steps_per_epoch',
        type=int,
        default=sys.maxsize,
        help=
        'Maximum number of training steps of effective batch size within an epoch to complete.'
    )

    parser.add_argument('--print_steps',
                        type=int,
                        default=100,
                        help='Interval to print training details.')

    parser.add_argument('--deepspeed_transformer_kernel',
                        default=False,
                        action='store_true',
                        help='Use DeepSpeed transformer kernel to accelerate.')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument(
        '--ckpt_type',
        type=str,
        default="DS",
        help="Checkpoint's type, DS - DeepSpeed, TF - Tensorflow, HF - Huggingface.")

    parser.add_argument(
        "--origin_bert_config_file",
        type=str,
        default=None,
        help="The config json file corresponding to the non-DeepSpeed pre-trained BERT model."
    )

    return parser


def get_summary_writer(name, base=".."):
    """Returns a tensorboard summary writer
    """
    return SummaryWriter(
        log_dir=os.path.join(base, SUMMARY_WRITER_DIR_NAME, name))


def write_summary_events(summary_writer, summary_events):
    for event in summary_events:
        summary_writer.add_scalar(event[0], event[1], event[2])


def is_time_to_exit(args, epoch_steps=0, global_steps=0):
    return (epoch_steps >= args.max_steps_per_epoch) or \
            (global_steps >= args.max_steps)


def check_early_exit_warning(args):
    # Issue warning if early exit from epoch is configured
    if args.max_steps < sys.maxsize:
        logging.warning(
            'Early training exit is set after {} global steps'.format(
                args.max_steps))

    if args.max_steps_per_epoch < sys.maxsize:
        logging.warning('Early epoch exit is set after {} global steps'.format(
            args.max_steps_per_epoch))
