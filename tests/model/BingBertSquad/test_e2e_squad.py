import subprocess as sp
import evaluate as eval
import datetime
import os
from math import isclose

def test_e2e_squad(tmpdir):
    squad_dir = "/data/BingBertSquad"
    base_dir = "../../../DeepSpeedExamples/BingBertSquad"

    script_name = "run_squad_deepspeed"
    model_file_name = "training_state_checkpoint_162.tar"
    eval_file_name = "dev-v1.1.json"

    num_gpus = 8
    timeout_sec = 5 * 60 * 60 # 5 hours
    expected_exact_match = 84.15
    expected_f1 = 90.95

    eval_version = "1.1"

    script = os.path.join(base_dir, script_name)
    model_file = os.path.join(squad_dir, model_file_name)
    eval_file = os.path.join(squad_dir, 

    # first test: deepspeed base
    output_dir = os.path.join(tmpdir, "DeepSpeed_Base")
    pred_file = os.path.join(output_dir, "/predictions.json")

    proc = sp.Popen(["bash", script, num_gpus, model_file, squad_dir, output_dir], stdout=sp.PIPE, stderr=sp.PIPE)

    try:
        outs, errs = proc.communicate(timeout=timeout_sec)

        if os.path.exists(pred_file):
            eval_result = eval.evaluate(eval_version, eval_file, pred_file)
            assert isclose(eval_result["exact_match"], expected_exact_match, abs_tol=1e-4)
            assert isclose(eval_result["f1"], expected_f1, abs_tol=1e-4)
        else:
            # error: what to do
            log_event("Error: Run Failed", outs, errs)


    except TimeoutExpired:
        proc.kill()
        # error: what to do
        log_event("Error: TimeOut", outs, errs)
    except subprocess.CalledProcessError: 
        # error: what to do
        log_event("Error: Run Failed", outs, errs)


    # second test: deepspeed with ZeRo-1 ...
    
    # third test: deepspeed with ZeRo-2 ...


def log_results(msg, outs, errs):
    # log.write(msg)
    # log.write(outs)
    # log.write(errs)


