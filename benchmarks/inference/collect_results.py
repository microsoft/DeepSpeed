import os
import re
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results-dir",
    "-r",
    type=str,
    default="./results",
    help="directory containing sweep results",
)
parser.add_argument("--version",
                    "-v",
                    type=int,
                    default=0,
                    help="version to be collected")
parser.add_argument("--gen-text-n",
                    "-n",
                    type=int,
                    default=1,
                    help="expected number of generated text")
parser.add_argument("--output",
                    "-o",
                    type=str,
                    default="./results.csv",
                    help="output file")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()


def get_branch(file_path):
    match = re.match(r".*\/(.*)\.log", file_path)
    if match is None:
        return ""
    else:
        return match.groups()[0]


def get_benchmark_params(root_dir, file_path):
    match = re.match(
        rf"{root_dir}\/(.+?)_(int8|fp16|fp32)_(true|false)_(\d+)gpus_v(\d+)\/",
        file_path,
    )
    if match is None:
        return {}
    else:
        model, dtype, graphs, gpus, version = match.groups()
        bool_dict = {"true": True, "false": False}
        return {
            "model": model,
            "model family": os.path.basename(root_dir),
            "dtype": dtype,
            "graphs": bool_dict[graphs.lower()],
            "gpus": int(gpus),
            "version": int(version),
        }


def get_perf_data(file_content):
    return_vals = {}
    latency_strs = re.findall(r"(latency\sstats\s.*\slatency\s[=]*[^=]*)", file_content)
    for latency_report in latency_strs:
        match = re.match(r"latency\sstats\s(.*)\s=", latency_report)
        if match is None:
            continue
        else:
            latency_type = match.groups()[0]
        matches = re.findall(r"\s+(.+?)\sLatency:\s+(\d+\.\d+)\sms", latency_report)
        return_vals.update({f"{latency_type} {key}": float(val) for key, val in matches})
    return return_vals


def get_generated_text(file_content, gen_text_n):
    return_vals = {}
    file_content = file_content.replace("\n", " ")
    file_content = file_content.replace("\t", " ")
    matches = re.findall(r"RESPONSE:\s+[-]{30}\s+(.+?)\s+[-]{30}", file_content)
    return_vals.update({f"generated-text": val for val in matches})
    return return_vals


def get_error(file_content):
    return_vals = {}
    matches = re.findall(r"(.*Error:\s+.+?)\n", file_content)
    return_vals.update({f"error": val for val in matches})
    return return_vals


if __name__ == "__main__":
    # List to collect data from all benchmarks
    benchmarks_data = []

    # Walk through directory of results from sweep.sh
    for root, dirs, files in os.walk(args.results_dir):
        # Because of how some models are named, the dir structure for results can vary, e.g.:
        # "EleutherAI/gpt-neo_*/baseline.log" versus "gpt2_*/baseline.log"
        if dirs:
            continue

        # Get data from baseline and each tested branch
        for name in files:
            file_path = os.path.join(root, name)

            branch = get_branch(file_path)
            if not branch and args.verbose:
                print(f"WARNING: Could not detect branch for file {file_path}, skipping")
                continue

            params = get_benchmark_params(args.results_dir, file_path)
            if not params and args.verbose:
                print(
                    f"WARNING: Could not detect benchmark settings for file {file_path}, skipping"
                )
                continue

            # Verify that the version matches that which we want to collect
            if params["version"] != args.version:
                continue

            with open(file_path, "r") as f:
                file_content = f.read()

            perf_data = get_perf_data(file_content)
            if not perf_data and args.verbose:
                print(
                    f"WARNING: Could not detect benchmark performance data for file {file_path}"
                )

            generated_text = get_generated_text(file_content, args.gen_text_n)
            if not generated_text and args.verbose:
                print(f"WARNING: Could not detect generated text for file {file_path}")

            error = get_error(file_content)
            if error and args.verbose:
                print(f"Error found in {file_path}, collecting error info...")

            benchmarks_data.append({
                "branch": branch,
                **params,
                **perf_data,
                **generated_text,
                **error,
            })

    # Convert to a DataFrame and save
    benchmarks_df = pd.DataFrame(benchmarks_data)
    print(benchmarks_df)
    benchmarks_df.to_csv(args.output)
