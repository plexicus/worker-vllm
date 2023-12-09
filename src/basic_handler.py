#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless worker. '''

# Start the vLLM serving layer on our RunPod worker.
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import runpod
import os
import logging

# Prepare the model and tokenizer
MODEL_NAME = os.environ.get('MODEL_NAME')
MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/runpod-volume/')
STREAMING = os.environ.get('STREAMING', False) == 'True'
TOKENIZER = os.environ.get('TOKENIZER', None)
USE_FULL_METRICS = os.environ.get('USE_FULL_METRICS', True)
QUANTIZATION = os.environ.get('QUANTIZATION', None)

if not TOKENIZER:
    TOKENIZER = None

if not MODEL_NAME:
    logging.error("Error: The model has not been provided.")

model_name_or_path = f"{MODEL_BASE_PATH}{MODEL_NAME.split('/')[1]}"
# To use a different branch, change revision
# For example: revision="gptq-4bit-128g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


async def handler(job: dict) -> dict[str, list]:
    '''
    This is the handler function that will be called by the serverless worker.
    '''
    logging.info("Job received by handler: {}".format(job))

    # Retrieve the job input.
    job_input = job['input']

    prompt = job_input['prompt']
    prompt_template=f'''You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
    ### Instruction:
    {prompt}
    ### Response:
    '''

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()

    output = model.generate(
        inputs=input_ids, 
        temperature=job_input.get("temperature", 0.7), 
        do_sample=True, 
        top_p=job_input.get("top_p", 0.95), 
        top_k=job_input.get("top_k", 40), 
        max_new_tokens=job_input.get("max_tokens", 40)
    )

    ret = {
        "text": tokenizer.decode(output[0])
    }
    return ret


# Start the serverless worker with appropriate settings
print("Starting the vLLM serverless worker with streaming disabled.")
runpod.serverless.start({
    "handler": handler
})
