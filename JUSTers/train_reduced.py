# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import csv
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange

device = torch.device("cuda")

from transformers import (
	WEIGHTS_NAME,
	AdamW,
	GPT2Config,
	GPT2LMHeadModel,
	GPT2Tokenizer,
	PreTrainedModel,
	PreTrainedTokenizer,
	get_linear_schedule_with_warmup,
)


try:
	from torch.utils.tensorboard import SummaryWriter
except ImportError:
	from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
	"gpt2-medium": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
	"gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}


class ComsenTextDataset(Dataset):
	def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
		assert os.path.isfile(file_path + "-x.csv")
		logger.info("Creating features from dataset file at %s", file_path)

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

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i):
		return torch.tensor(self.examples[i])


def load_and_cache_examples(args, tokenizer, evaluate=False):
	file_path = args.train_data_file
	return ComsenTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)



def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
	""" Train the model """

	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

	def collate(examples: List[torch.Tensor]):
		if tokenizer._pad_token is None:
			return pad_sequence(examples, batch_first=True)
		return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

	train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
	train_dataloader = DataLoader(
		train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
	)

	if args.max_steps > 0:
		t_total = args.max_steps
		args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

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
	
	
	# Distributed training (should be after apex fp16 initialization)
	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
		)

	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info(
		"  Total train batch size (w. parallel, distributed & accumulation) = %d",
		args.train_batch_size
		* args.gradient_accumulation_steps
		* (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
	)
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	epochs_trained = 0
	steps_trained_in_current_epoch = 0

	tr_loss, logging_loss = 0.0, 0.0

	model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
	model_to_resize.resize_token_embeddings(len(tokenizer))

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

			inputs, labels =  (batch, batch)
			inputs = inputs.to(args.device)
			labels = labels.to(args.device)
			model.train()
			outputs = model(inputs, labels=labels)
			loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps
			else:
				loss.backward()

			tr_loss += loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1
			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break
		if args.do_eval:
			with open('data_dir/development-x.csv', 'r') as file:
				lines = list(csv.reader(file))
			lines = lines[1:]
			reasons_eval = list()
			for line in lines:
				encoded_prompt = tokenizer.encode(line[1].strip() + ' <|continue|>', add_special_tokens=False, return_tensors="pt")
				encoded_prompt = encoded_prompt.to(args.device)
				output_sequences = model.generate(
						input_ids=encoded_prompt,
						do_sample=True,
						max_length=128,
						temperature=1.0,
						top_k=50,
						top_p=0.9,
						repetition_penalty=1.0)

				generated_sequence = output_sequences[0].tolist()
				text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
				reasons_eval.append(text.split('<|continue|>')[1].strip().replace('!',''))
			from bert_score import score as BERT_scorer_hugging
			#self.references_df = pd.DataFrame()
			#self.references_df = pd.read_csv(reference_path, index_col = 0, names=['ref1','ref2','ref3'])
			references_array = pd.read_csv('data_dir/development-y.csv', index_col = 0, names=['ref1','ref2','ref3']).to_numpy()
			print(len(reasons_eval))
			print(len(references_array.tolist()))
			print(np.mean(BERT_scorer_hugging(reasons_eval,references_array.tolist(), lang = "en", rescale_with_baseline=True)[2].numpy()))
		exit()
		if args.max_steps > 0 and global_step > args.max_steps:
			train_iterator.close()
			break


	return global_step, tr_loss / global_step



def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument(
		"--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		required=True,
		help="The output directory where the model predictions and checkpoints will be written.",
	)
	parser.add_argument(
		"--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
	)

	# Other parameters
	parser.add_argument(
		"--eval_data_file",
		default=None,
		type=str,
		help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
	)
	parser.add_argument(
		"--line_by_line",
		action="store_true",
		help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
	)
	parser.add_argument(
		"--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
	)
	parser.add_argument(
		"--model_name_or_path",
		default=None,
		type=str,
		help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
	)

	parser.add_argument(
		"--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
	)
	parser.add_argument(
		"--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
	)

	parser.add_argument(
		"--config_name",
		default=None,
		type=str,
		help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
	)
	parser.add_argument(
		"--tokenizer_name",
		default=None,
		type=str,
		help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
	)
	parser.add_argument(
		"--cache_dir",
		default=None,
		type=str,
		help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
	)
	parser.add_argument(
		"--block_size",
		default=-1,
		type=int,
		help="Optional input sequence length after tokenization."
		"The training dataset will be truncated in block of this size for training."
		"Default to the model max input length for single sentence inputs (take into account special tokens).",
	)
	parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
	parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
	parser.add_argument(
		"--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
	)

	parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
	parser.add_argument(
		"--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
	)
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument(
		"--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
	)
	parser.add_argument(
		"--max_steps",
		default=-1,
		type=int,
		help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
	)
	parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

	parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
	parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
	parser.add_argument(
		"--save_total_limit",
		type=int,
		default=None,
		help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
	)
	parser.add_argument(
		"--eval_all_checkpoints",
		action="store_true",
		help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
	)
	parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
	parser.add_argument(
		"--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
	)
	parser.add_argument(
		"--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
	)
	parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
	parser.add_argument(
		"--fp16",
		action="store_true",
		help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
	)
	parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
	args = parser.parse_args()

	if (
		os.path.exists(args.output_dir)
		and os.listdir(args.output_dir)
		and args.do_train
		and not args.overwrite_output_dir
	):
		raise ValueError(
			"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
				args.output_dir
			)
		)

	# Setup CUDA, GPU & distributed training
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend="nccl")
		args.n_gpu = 1
	args.device = device

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
	)
	logger.warning(
		"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
		args.local_rank,
		device,
		args.n_gpu,
		bool(args.local_rank != -1),
		args.fp16,
	)

	# Set seed
	set_seed(args)

	# Load pretrained model and tokenizer
	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

	config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
	device = torch.device("cuda")
	if args.config_name:
		config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
	elif args.model_name_or_path:
		config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
	else:
		config = config_class()

	if args.tokenizer_name:
		tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
	elif args.model_name_or_path:
		tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
	else:
		raise ValueError(
			"You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
			"and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
		)

	if args.block_size <= 0:
		args.block_size = tokenizer.max_len_single_sentence
		# Our input block size will be the max possible for the model
	else:
		args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

	if args.model_name_or_path:
		model = model_class.from_pretrained(
			args.model_name_or_path,
			from_tf=bool(".ckpt" in args.model_name_or_path),
			config=config,
			cache_dir=args.cache_dir,
		)
	else:
		logger.info("Training new model from scratch")
		model = model_class(config=config)

	model.to(args.device)

	if args.local_rank == 0:
		torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

	logger.info("Training/evaluation parameters %s", args)

	# Training
	if args.do_train:
		if args.local_rank not in [-1, 0]:
			torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

		train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

		if args.local_rank == 0:
			torch.distributed.barrier()

		global_step, tr_loss = train(args, train_dataset, model, tokenizer)
		logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
	main()
