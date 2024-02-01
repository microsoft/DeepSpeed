import mii


def fake_request_texts(batch_size: int):
    request_texts = ["Ha ha ha"] * batch_size
    return request_texts


if __name__ == '__main__':
    model_id = "meta-llama/Llama-2-7b-hf"

    batch_size = 32
    prompts = fake_request_texts(batch_size)

    pipe = mii.pipeline(model_name_or_path=model_id, quantization_mode='wf6af16')
    response = pipe(prompts, max_new_tokens=2)
    print(f"{len(response)} responses.")
