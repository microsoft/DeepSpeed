import mii
# It requires to install deepspeed_mii: pip install deepspeed-mii


def fake_request_texts(batch_size : int):
  request_texts = ["DeepSpeed is "] * batch_size
  return request_texts


def run_pipeline(deployment_name : str):
  pipe = mii.pipeline(model_name_or_path = deployment_name)
  batch_size = 4
  request_texts = fake_request_texts(batch_size)
  response = pipe(request_texts, max_new_tokens=128)
  print(f"{len(response)} responses.")
  # print(f"Response: {response}.")


if __name__ == "__main__":
  deployment_name = "meta-llama/Llama-2-7b-hf"
  run_pipeline(deployment_name)
