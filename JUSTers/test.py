from transformers import AutoTokenizer, AutoModel  

tokenizer = AutoTokenizer.from_pretrained("aliosm/ComVE-gpt2-medium")  
model = AutoModel.from_pretrained("aliosm/ComVE-gpt2-medium") 
print('success')

prompt_text = 'I drink an apple'

encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
print(encoded_prompt)
output_sequences = model(encoded_prompt)
generated_sequence = output_sequences[0].tolist()
text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
print(text)
print('success')