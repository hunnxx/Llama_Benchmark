import json
import argparse

from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import MllamaProcessor, MllamaForConditionalGeneration

from utils import DocVQADataset

parser = argparse.ArgumentParser()
parser.add_argument("--bench_type", type=str, default="docqa", help="Type of benchmark")
parser.add_argument("--device", type=str, default="auto", help="cuda for gpu | auto for multiple gpu")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--dtype", type=str, default='float32', help="float32 | bfloat16")
parser.add_argument("--access_token", type=str, default=None, help="Huggingface access token")
args = parser.parse_args()


def preprocessing_data(img_paths, qs, q_ids):
    return_imgs = []
    return_qs = []
    return_q_ids = []
    for img_path, q, q_id in zip(img_paths, qs, q_ids):
        img = Image.open(img_path)
        input_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q}
                ]
            }
        ]
        
        return_imgs.append(img)
        return_qs.append(input_prompt)
        return_q_ids.append(q_id)
        
    return return_imgs, return_qs, return_q_ids


def postprocessing_data(data):
    return data.split('assistant\n\n')[-1]
    

def prepare_model(args):
    if args.device not in ('cuda', 'cuda:0', "cuda:1", "cuda:2", "cuda:3", "auto"):
        raise Exception('device map')
    if args.batch_size < 1:
        raise Exception('batch size')
    if args.dtype not in ('float32', "bfloat16"):
        raise Exception('dtype')
    if args.access_token is None:
        raise Exception('huggingface access token')
    
    __model_name_or_path = 'Llama-3.2-11B-Vision-Instruct'
    __device_map = args.device
    __dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    __access_token = args.access_token
    
    __processor = MllamaProcessor.from_pretrained(f"meta-llama/{__model_name_or_path}", padding_side="left", token=__access_token)
    __model = MllamaForConditionalGeneration.from_pretrained(f"meta-llama/{__model_name_or_path}", torch_dtype=__dtype, device_map=__device_map, token=__access_token)

    return __processor, __model


def main():
    docvqa_dataset = DocVQADataset(file_path='data/docvqa/test.jsonl')
    docvqa_dataloader = DataLoader(docvqa_dataset, batch_size=args.batch_size, shuffle=False)

    processor, model = prepare_model(args)

    results = []
    for img_paths, qs, q_ids in tqdm(docvqa_dataloader):
        imgs, input_prompts, input_q_ids = preprocessing_data(img_paths, qs, q_ids)
        input_prompts = processor.apply_chat_template(input_prompts, add_generation_prompt=True)
        inputs = processor(text=input_prompts, images=imgs, padding=True, return_tensors='pt').to(model.device, dtype=model.dtype)

        output = model.generate(**inputs, max_new_tokens=100)
        gen_ids = output
        gen_output = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        for gen_idx, gen_out in enumerate(gen_output):
            answer = postprocessing_data(gen_out)
            results.append({"questionId": input_q_ids[gen_idx].item(), "answer": answer})

    with open('data/docvqa/test_results.json', 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    main()