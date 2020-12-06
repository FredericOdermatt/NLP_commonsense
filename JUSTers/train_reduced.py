"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.train()

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

text_batch = ["I love Pixar.", "I don't care for Pixar."]
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
print(encoding)

input_ids = encoding['input_ids']
print(input_ids)
attention_mask = encoding['attention_mask']
print(attention_mask)

'''
labels = torch.tensor([1,0]).unsqueeze(0)
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
'''

'''
import os
import csv

import numpy as np
import torch

from transformers import (
	AdamW,
	GPT2Config,
	GPT2LMHeadModel,
	GPT2Tokenizer,
	PreTrainedModel,
	PreTrainedTokenizer
)

from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('bert-base-uncased')
model.train()

adam_learning_rate = 5e-5 ## default starting learning rate in JUSTer
adam_epsilon = 1e-8 ## default Epsilon for Adam optimizer in JUSTer

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(model.parameters(), lr=adam_learning_rate, eps=adam_epsilon)

# Load pretrained model and tokenizer
config = config_class.from_pretrained('GPT2Config', cache_dir=args.cache_dir) ??
tokenizer = tokenizer_class.from_pretrained('GPT2Tokenizer', cache_dir=args.cache_dir) ??
block_size = 128 ??

model = model_class.from_pretrained(
		args.model_name_or_path,
		from_tf=bool(".ckpt" in args.model_name_or_path),
		config=config,
		cache_dir=args.cache_dir)

model.to(args.device) ??

X_lines = list()
with open(file_path + "-x.csv") as f:
	reader = csv.reader(f)

	for row in reader:
		X_lines.append(row[1])
X_lines = X_lines[1:]

Y_lines = list()
with open(file_path + "-y.csv") as f:
	reader = csv.reader(f)

	for row in reader:
		Y_lines.append(row[1:])

lines = list()
for x_line, y_line in zip(X_lines, Y_lines):
	for i in range(3):
		if len(y_line[i].strip()) > 0:
			lines.append(x_line.strip() + " <|continue|> " + y_line[i].strip())
random.shuffle(lines)

self.examples = tokenizer.batch_encode_plus(lines, max_length=block_size)["input_ids"]

global_step, tr_loss = train(args, train_dataset, model, tokenizer)
""" Train the model """



train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
train_dataloader = DataLoader(
	train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
	{
		"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
		"weight_decay": args.weight_decay,
	},
	{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
	optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
)

# Check if saved optimizer or scheduler states exist
if (
	args.model_name_or_path
	and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
	and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
):
	# Load in optimizer and scheduler states
	optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
	scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

global_step = 0
epochs_trained = 0
steps_trained_in_current_epoch = 0

tr_loss, logging_loss = 0.0, 0.0

model.zero_grad()
train_iterator = trange(
	epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
)
set_seed(args)  # Added here for reproducibility
for _ in train_iterator:
	epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
	for step, batch in enumerate(epoch_iterator):

		# Skip past any already trained steps if resuming training
		if steps_trained_in_current_epoch > 0:
			steps_trained_in_current_epoch -= 1
			continue

		inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
		inputs = inputs.to(args.device)
		labels = labels.to(args.device)
		model.train()
		outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
		loss = outputs[0]  # model outputs are always tuple in transformers (see doc)


		tr_loss += loss.item()
		if (step + 1) % args.gradient_accumulation_steps == 0:
			optimizer.step()
			scheduler.step()  # Update learning rate schedule
			model.zero_grad()
			global_step += 1

'''