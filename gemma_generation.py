import argparse

import torch
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline



if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("-c", "--checkpoint", type=int, \
      default=0,\
      help="index for model checkpoint "\
      "0: HuggingFaceH4/zephyr-7b-gemma-v0.1"\
      "1: HuggingFaceH4/zephyr-7b-gemma-sft-v0.1"\
      "2: SakanaAI/DiscoPOP-zephyr-7b-gemma")
  parser.add_argument("-e", "--example_type", type=int, \
      default=0,\
      help="index for type of examples to load "\
      "1: No examples (zero-shot)"\
      "2: Plain examples (few-shot)"
      "3: Chain-of-thought examples (few-shot with c.o.t.)"
      "4: Chain-of-thought examples plus 'Let's think step-by-step: '"
      "5: few-shot + 'Let's think step-by-step: ' appended after 'A: '"
      "0: use all examples"\
      )
  parser.add_argument("-q", "--question", type=str, \
      default = "Roger has 5 tennis balls. He buys 2 more cans of tennis balls."\
      " Each can has 3 tennis balls. How many tennis balls does he have now?",\
      help="A question, i.e. an arithmetic or simple maths word problem")


  args = parser.parse_args()
  question = args.question
  checkpoint_index = args.checkpoint
  example_type = args.example_type

  checkpoints = [\
      "HuggingFaceH4/zephyr-7b-gemma-v0.1",\
      "HuggingFaceH4/zephyr-7b-gemma-sft-v0.1",\
      "SakanaAI/DiscoPOP-zephyr-7b-gemma"]

  checkpoint = checkpoints[checkpoint_index] 

  pipe = pipeline("text-generation", model=checkpoint, device_map="auto", torch_dtype=torch.bfloat16)

  query_response = True

  examples_list = []

  if example_type == 1 or example_type == 0:
    examples_list.append("")
  if example_type == 2 or example_type == 0:
    with open("few_shot_examples.txt", "r") as f:
      examples_list.append(f.read())
  if example_type == 3 or example_type == 0:
    with open("cot_examples.txt", "r") as f:
      examples_list.append(f.read())
  if example_type == 4 or example_type == 0:
    with open("cot_step_by_step_examples.txt", "r") as f:
      examples_list.append(f.read())
  if example_type == 5 or example_type == 0:
    with open("step_by_step_examples.txt", "r") as f:
      examples_list.append(f.read())

  while query_response:

    query_response = False
    # queries include questin poised as standard zero-shot/few-shot, with chain-of-though examples, 
    # chain-of-thought plus 'Let's think step-by-step', and 'let's think step-by-step:' alone

    for examples in examples_list:
      queries = [\
          {"role": "user", "content": f"{examples} Q: {question}\nA: "},\
          ]
        
      t0 = time.time(); 
      outputs = pipe(\
          queries,\
          max_new_tokens=64,\
          do_sample=True,\
          temperature=0.7,\
          top_k=50,\
          top_p=0.95,\
          stop_sequence="<|im_end|>",\
          )
      t1 = time.time(); 

      msg = f"generation with {checkpoint} on {pipe.device} w/ {pipe.torch_dtype} in {t1-t0:.3f} seconds\n"

      msg += f"response:\n\n\t\t {outputs[0]['generated_text'][-1]['content']}"

      print(msg)

    question = input("Enter another question (leave blank to end session)")
    if question == "":
      pass
    else:
      query_response = True
