from mii import pipeline
# It requires to install deepspeed_mii: pip install deepspeed-mii


def fake_request_texts(batch_size : int):
  request_texts = ["is a "] * batch_size
  return request_texts


def run_pipeline(deployment_name : str):
  pipe = pipeline(model_name_or_path = deployment_name)
  batch_size = 2
  request_texts = fake_request_texts(batch_size)
  response = pipe(request_texts, max_new_tokens=128)
  print(f"Generated {len(response.generated_texts)} responses.")


if __name__ == "__main__":
  # deployment_name = "meta-llama/Llama-2-7b-hf"
  deployment_name = "facebook/opt-1.3b"
  run_pipeline(deployment_name)