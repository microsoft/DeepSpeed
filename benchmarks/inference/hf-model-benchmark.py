import os
import time
import argparse
import torch
import deepspeed

from transformers import pipeline
from huggingface_hub import HfApi


def torch_dtype(value):
    if value == "int8":
        return torch.int8
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    else:
        raise NotImplementedError(f"Unknown datatype: {value}")


def print_latency(latency_set, title):
    # trim warmup queries
    latency_set = list(latency_set)
    count = len(latency_set)
    if count > 0:
        latency_set.sort()
        n50 = (count - 1) * 0.5 + 1
        n90 = (count - 1) * 0.9 + 1
        n95 = (count - 1) * 0.95 + 1
        n99 = (count - 1) * 0.99 + 1
        n999 = (count - 1) * 0.999 + 1

        avg = sum(latency_set) / count
        p50 = latency_set[int(n50) - 1]
        p90 = latency_set[int(n90) - 1]
        p95 = latency_set[int(n95) - 1]
        p99 = latency_set[int(n99) - 1]
        p999 = latency_set[int(n999) - 1]

        print(f"====== latency stats {title} ======")
        print("\tAvg Latency: {0:8.2f} ms".format(avg))
        print("\tP50 Latency: {0:8.2f} ms".format(p50))
        print("\tP90 Latency: {0:8.2f} ms".format(p90))
        print("\tP95 Latency: {0:8.2f} ms".format(p95))
        print("\tP99 Latency: {0:8.2f} ms".format(p99))
        print("\t999 Latency: {0:8.2f} ms".format(p999))


def get_task_query(args):
    task, model = args.task, args.model
    if task == "text-generation":
        return "DeepSpeed is"
    if task == "fill-mask":
        if "roberta" in model:
            return "Hello I'm a <mask> model"
        else:
            return "Hello I'm a [MASK] model"
    else:
        raise NotImplementedError(f"Task not recognized: {task}")


def get_task_kwargs(args):
    task = args.task
    if task == "text-generation":
        return {
            "min_length": args.tokens,
            "max_new_tokens": args.tokens,
            "do_sample": False,
        }
    else:
        return {}


def get_response(args, response):
    task = args.task
    if task == "text-generation":
        return response[0]["generated_text"]
    if task == "fill-mask":
        return [r["token_str"] for r in response]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="hf model name")
    parser.add_argument(
        "--dtype",
        type=torch_dtype,
        default="fp16",
        help="int8, fp16, or fp32",
    )
    parser.add_argument("--graph", action="store_true", help="CUDA Graphs on")
    parser.add_argument("--kernel-inject",
                        action="store_true",
                        help="inject DeepSpeed kernels")
    parser.add_argument(
        "--tokens",
        type=int,
        default=50,
        help="tokens to generate for text-generation task",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.getenv("LOCAL_RANK",
                              "0")),
        help="local rank",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=int(os.getenv("WORLD_SIZE",
                              "1")),
        help="world size",
    )
    parser.add_argument("--trials", type=int, default=30, help="number of trials")
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="number of trials to discard at beginning of measurement",
    )
    args = parser.parse_args()

    # Determine the task for a model
    args.task = HfApi().model_info(args.model).pipeline_tag

    # Initialize DeepSpeed
    deepspeed.init_distributed("nccl")

    # Check for valid settings
    if args.world_size > 1:
        assert args.kernel_inject, "MP size >1 requires DeepSpeed kernels"
        assert not args.graph, "CUDA Graphs cannot be used with MP size >1"
    if args.graph:
        assert args.kernel_inject, "DeepSpeed kernels must be used with CUDA Graphs"
    if args.dtype == torch.int8 and not args.kernel:
        print("WARNING: DeepSpeed kernels must be used for int8")
        print("Reverting dtype to fp16")
        args.dtype = torch.float16

    # Print out arguments
    if args.local_rank == 0:
        print("BENCHMARK SETTINGS:")
        print(f"\tMODEL: {args.model}")
        print(f"\tTASK: {args.task}")
        print(f"\tTOKENS: {args.tokens} (text-generation only)")
        print(f"\tDTYPE: {args.dtype}")
        print(f"\tCUDA_GRAPHS: {args.graph}")
        print(f"\tKERNEL_INJECT: {args.kernel_inject}")

    # Load the HF pipeline
    pipe = pipeline(args.task, model=args.model, framework="pt", device=args.local_rank)

    # Convert to half (this may not be necessary)
    if args.dtype == torch.float16:
        pipe.model = pipe.model.half()

    # Load DS inference engine
    pipe.model = deepspeed.init_inference(
        pipe.model,
        dtype=args.dtype,
        replace_with_kernel_inject=args.kernel_inject,
        replace_method="auto",
        enable_cuda_graph=args.graph,
    )

    # Get query and kwargs based on input parameters
    query = get_task_query(args)
    kwargs = get_task_kwargs(args)

    # Determine exact number of tokens generated for text-generation task
    if args.task == "text-generation":
        input_tokens = pipe.preprocess(query)
        input_token_count = input_tokens["input_ids"].shape[-1]
        output_token_count = pipe.forward(input_tokens,
                                          **kwargs)["generated_sequence"].shape[-1]
        generated_token_count = output_token_count - input_token_count
    else:
        generated_token_count = 1

    # Measure performance over several trials
    e2e_times = []
    model_times = []
    pipe.model.profile_model_time()
    for i in range(args.trials):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()

        response = pipe(query, **kwargs)

        torch.cuda.synchronize()
        end = time.perf_counter_ns()

        e2e_t = (end - start) / (1e6 * generated_token_count)  # Convert ns to ms
        model_t = pipe.model.model_times()
        if i >= args.warmup:
            e2e_times.append(e2e_t)
            model_times.extend(model_t)

    # Print out performance results and single response
    if args.local_rank == 0:
        print_latency(e2e_times, "e2e latency")
        print_latency(model_times, "model_latency")
        print("RESPONSE:")
        print("-" * 30)
        print(get_response(args, response))
        print("-" * 30)
