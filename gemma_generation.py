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
      "6: zero-shot + 'Let's think step-by-step: ' appended after 'A: '"
      "7: zero-shot + 'The answer is: ' appended after 'A: '"
      "0: use all examples"\
      )
  parser.add_argument("-l", "--log_path", type=str, \
      default="",\
      help="filepath to log results to (leave blank to avoid logging)")
  parser.add_argument("-n", "--no_loop", action="store_false", \
      default=True,\
      help="")
  parser.add_argument("-q", "--question", type=str, \
      default = "Roger has 5 tennis balls. He buys 2 more cans of tennis balls."\
      " Each can has 3 tennis balls. How many tennis balls does he have now?",\
      help="A question, i.e. an arithmetic or simple maths word problem")


  args = parser.parse_args()
  question = args.question
  checkpoint_index = args.checkpoint
  example_type = args.example_type
  query_response = 1 * args.no_loop + 1
  do_log = args.log_path != ""
  
  verbose = False

  checkpoints = [\
      "HuggingFaceH4/zephyr-7b-gemma-v0.1",\
      "HuggingFaceH4/zephyr-7b-gemma-sft-v0.1",\
      "SakanaAI/DiscoPOP-zephyr-7b-gemma"]

  checkpoint = checkpoints[checkpoint_index] 

  pipe = pipeline("text-generation", model=checkpoint, device_map="auto", torch_dtype=torch.bfloat16)


  examples_list = []
  example_types = []

  if example_type == 1 or example_type == 0:
    examples_list.append("")
    example_types.append(1)
  if example_type == 2 or example_type == 0:
    with open("few_shot_examples.txt", "r") as f:
      examples_list.append(f.read())
    example_types.append(2)
  if example_type == 3 or example_type == 0:
    with open("cot_examples.txt", "r") as f:
      examples_list.append(f.read())
    example_types.append(3)
  if example_type == 4 or example_type == 0:
    with open("cot_step_by_step_examples.txt", "r") as f:
      examples_list.append(f.read())
    example_types.append(4)
  if example_type == 5 or example_type == 0:
    with open("step_by_step_examples.txt", "r") as f:
      examples_list.append(f.read())
    example_types.append(5)
  if example_type == 6 or example_type == 0:
    examples_list.append("")
    example_types.append(6)
  if example_type == 7 or example_type == 0:
    examples_list.append("Respond with only the answer using as short an answer as possible.")
    example_types.append(7)

  example_type_dict = {1: "zero-shot",\
      2: "few-shot",\
      3: "chain-of-thought",\
      4: "chain-of-thought + 'Let's think step-by-step'",\
      5: "few-shot + 'Let's think step-by-step'",\
      6: "zero-shot + 'Let's think step-by-step'",\
      7: "zero-shot + 'The answer is:'"\
      }

  while query_response:

    if do_log:
      log_results = ""

    query_response -= 1
    # queries include questin poised as standard zero-shot/few-shot, with chain-of-though examples, 
    # chain-of-thought plus 'Let's think step-by-step', and 'let's think step-by-step:' alone

    for example_type, examples in zip(example_types, examples_list):

      if example_type == 4 or example_type == 5 or example_type == 6:
        answer_prepend = "Let's think step-by-step: "
      elif example_type == 7:
        answer_prepend = "The answer is: "
      else:
        answer_prepend = ""

      queries = [\
          {"role": "user", "content": f"{examples} Q: {question}\nA: {answer_prepend}"},\
          ]
        
      t0 = time.time(); 
      outputs = pipe(\
          queries,\
          max_new_tokens=512,\
          do_sample=True,\
          temperature=0.7,\
          top_k=50,\
          top_p=0.95,\
          stop_sequence="<|im_end|>",\
          )
      t1 = time.time(); 


      msg = f"\n\n\tQuestion:\n\n{question}\n"
    
      if verbose: 
        msg += f"\ngeneration with {checkpoint} on {pipe.device} w/ {pipe.torch_dtype} in {t1-t0:.3f} seconds\n"
        msg += f"\n\tPrompt:\n {outputs[0]['generated_text'][0]['content']}"

      msg += f"\n\t{example_type_dict[example_type]} Response:\n\n"
      msg += f"{outputs[0]['generated_text'][-1]['content']}"
      
      if do_log:
        log_results = f"\n\n\tQuestion:\n\n{question}\n"
      
        log_results += f"\ngeneration with {checkpoint} on {pipe.device} w/ {pipe.torch_dtype} in {t1-t0:.3f} seconds\n"
        log_results += f"\n\tPrompt:\n {outputs[0]['generated_text'][0]['content']}"

        log_results += f"\n\t{example_type_dict[example_type]} Response:\n\n"
        log_results += f"{outputs[0]['generated_text'][-1]['content']}"

        with open(args.log_path, "a") as f:

          f.write(log_results)

      print(msg)

      #print("\n", outputs[0], "\n")


    if query_response:
      question = input("Enter another question (leave blank to end session)")
      if question == "":
        query_response = False
      else:
        query_response = 2

