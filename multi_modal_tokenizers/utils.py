import json
from huggingface_hub import hf_hub_download
from safetensors import safe_open

def load_state_from_repo(repo_id):
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    with open(config_path, 'r') as file:
        config = json.load(file)
    state_dict = {}
    with safe_open(model_path, framework="pt") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    return state_dict, config
