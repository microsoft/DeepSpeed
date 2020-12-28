"""
example usage:
python scripts/run_gpt2_eval.py \
  --model-parallel-size 1 \
  --num-layers 12 \
  --hidden-size 768 \
  --num-attention-heads 12 \
  --model-path <gpt2_117_path> \
  --data-path <wikitext_tokens_test_path> \
  --batch-size 16 \
  --cache-dir <cache dir path>
"""
import argparse
import subprocess

parser = argparse.ArgumentParser('run zero shot GPT2 eval')
parser.add_argument('--model-path', type=str, required=True,
                    help='Saved model path for evaluation')
parser.add_argument('--batch-size', type=int, default=4,
                    help='batch size to use for evaluation')
parser.add_argument('--num-attention-heads', type=int, default=12,
                    help='num of transformer attention heads')
parser.add_argument('--hidden-size', type=int, default=768,
                    help='tansformer hidden size')
parser.add_argument('--num-layers', type=int, default=12,
                    help='num decoder layers')
parser.add_argument('--data-path', type=str, required=True,
                    help='Data path for evaluation data')
parser.add_argument('--cloze-eval', action='store_true',
                    help='Run lambada cloze eval instead of perplexity eval.')
parser.add_argument('--webtext-eval', action='store_true',
                    help='Run webtext PPL eval instead of wikitext PPL eval.')
parser.add_argument('--eval-iters', default=5000, type=int,
                    help='number of iterations to run webtext evaluation')
parser.add_argument('--model-parallel-size', type=int, default=1,
                    help='model parallel size to use')
parser.add_argument('--load-openai', action='store_true',
                    help='Load weights from saved openai/hf checkpoints')
parser.add_argument('--cache-dir', type=str, default='cache',
                    help='directory to cache gpt2 tokenizers')
args = parser.parse_args()

multinode_args = ''
if args.model_parallel_size > 1:
    multinode_args += ' -m torch.distributed.launch --nproc_per_node {} '.format(args.model_parallel_size)

CMD = ' --model-parallel-size {model_par} \
       --num-layers {nlayers} \
       --hidden-size {hidden} \
       --log-interval 100 \
       --load {model} \
       --eval-batch-size {batch} \
       --num-attention-heads {natt} \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --text-key text \
       --distributed-backend nccl \
       --hidden-dropout 0.1 \
       --attention-dropout 0.1 \
       --fp16 \
       --overlapping-eval 32 \
       --cache-dir {cache} '.format(model_par=args.model_parallel_size,
                                    nlayers=args.num_layers,
                                    hidden=args.hidden_size,
                                    model=args.model_path,
                                    batch=args.batch_size,
                                    natt=args.num_attention_heads,
                                    cache=args.cache_dir)

if args.load_openai:
    CMD += ' --load-openai '
if args.cloze_eval:
    CMD += ' --cloze-eval '
    CMD = 'evaluate_gpt2.py' + CMD
    print('Running Lambada Eval Command:', flush=True)
elif args.webtext_eval:
    CMD += '--train-iters 0 --eval-iters {} --test-data {} --loose-json '.format(args.eval_iters, args.data_path)
    CMD = 'pretrain_gpt2.py' + CMD
    print('Running Webtext Eval Command:', flush=True)
else:
    CMD += ' --valid-data {} '.format(args.data_path)
    CMD = 'evaluate_gpt2.py' + CMD
    print('Running PPL Eval Command:', flush=True)

CMD = 'python3 '+multinode_args+CMD
print(CMD, flush=True)

subprocess.call(CMD.split())
