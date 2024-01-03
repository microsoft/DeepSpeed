import mii
import os
# It requires to install deepspeed_mii: pip install deepspeed-mii


def fake_request_texts(batch_size : int):
  request_texts = ["Ha ha ha"] * batch_size
  return request_texts


def run_pipeline(deployment_name : str):
  pipe = mii.pipeline(model_name_or_path = deployment_name)
  batch_size = 32
  request_texts = fake_request_texts(batch_size)
  response = pipe(request_texts, max_new_tokens=128)
  print(f"{len(response)} responses.")
  # print(f"Response: {response}.")


if __name__ == "__main__":
  # current directory: debug
  # deployment_name = f"{os.path.dirname(__file__)}/models--princeton-nlp--Sheared-LLaMA-1.3B"
  # deployment_name = f"{os.path.dirname(__file__)}/priceton-nlp/Sheared-LLaMA-1.3B-processed"
  # deployment_name = "Qwen/Qwen-1_8B-Chat-Int4"
  deployment_name = f"{os.path.dirname(__file__)}/models--princeton-nlp--Sheared-LLaMA-1.3B/snapshots/dbdd43e8018c14ec50f2f569564b328b28f254b9"
  run_pipeline(deployment_name)
