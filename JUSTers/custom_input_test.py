### might transformers version 2.4.1, torch version 1.18.0, tensorflow some old version
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
device = torch.device("cuda")
#tokenizer = AutoTokenizer.from_pretrained("aliosm/ComVE-gpt2-medium")  
tokenizer = AutoTokenizer.from_pretrained("./models_dir/gpt2/")  
#model = AutoModelWithLMHead.from_pretrained("aliosm/ComVE-gpt2-medium") 
model = AutoModelWithLMHead.from_pretrained("./models_dir/gpt2/") 


custom_input = ['Summer in North America is great for skiing,  snowshoeing,  and making a snowman.']

prompt_text = custom_input[0] + ' <|continue|>'
encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=128,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.0
        )
generated_sequence = output_sequences[0].tolist()
text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

print(custom_input[0])
print(text.split('<|continue|>')[1].strip())

'''
import csv
with open('dat.csv', 'r') as file:
    lines = list(csv.reader(file))
lines = lines[1:]

for line in lines:
    print(line)
    prompt_text = line[1].strip() + ' <|continue|>'
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    output_sequences = model.generate(
            input_ids=encoded_prompt,
            do_sample=True,
            max_length=128,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.0
        )
    
    # Batch size == 1. to add more examples please use num_return_sequences > 1
    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    print(text.split('<|continue|>')[1].strip())
'''
