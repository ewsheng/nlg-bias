"""pre/post processing functions."""


import logging
import numpy as np
import os

from constants import *

logger = logging.getLogger(__name__)


class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, words, label=None):
		"""Constructs a InputExample.

		Args:
		  guid: Unique id for the example.
		  words: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
		  label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.words = words
		self.label = label


class InputFeatures(object):
	"""A single set of features of data."""
	def __init__(self,
				   input_ids,
				   input_mask,
				   segment_ids,
				   label_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id


def read_examples_from_file(data_dir, data_file, is_test=False):
	file_path = os.path.join(data_dir, data_file)
	guid_index = 1
	examples = []
	with open(file_path, encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			splits = line.split('\t')
			words = splits[-1].split()
			if not is_test:
				label = int(splits[0])
			else:
				label = 0
			examples.append(InputExample(guid="%s-%d".format(data_file, guid_index),
										 words=words,
										 label=label))
	return examples


def convert_examples_to_features(examples,
								 label_list,
								 max_seq_length,
								 tokenizer,
								 cls_token_at_end=False,
								 cls_token="[CLS]",
								 cls_token_segment_id=1,
								 sep_token="[SEP]",
								 sep_token_extra=False,
								 pad_on_left=False,
								 pad_token=0,
								 pad_token_segment_id=0,
								 pad_token_label_id=-1,
								 sequence_a_segment_id=0,
								 mask_padding_with_zero=True):
	""" Loads a data file into a list of `InputBatch`s
		`cls_token_at_end` define the location of the CLS token:
			- False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
			- True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
		`cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
	"""

	label_map = {label: i for i, label in enumerate(label_list)}

	features = []
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d", ex_index, len(examples))

		tokens = []
		for word in example.words:
			word_tokens = tokenizer.tokenize(word)
			tokens.extend(word_tokens)
		label_id = label_map[example.label]

		# Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
		special_tokens_count = 3 if sep_token_extra else 2
		if len(tokens) > max_seq_length - special_tokens_count:
			tokens = tokens[:(max_seq_length - special_tokens_count)]

		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids:   0   0   0   0  0     0   0
		#
		# Where "type_ids" are used to indicate whether this is the first
		# sequence or the second sequence. The embedding vectors for `type=0` and
		# `type=1` were learned during pre-training and are added to the wordpiece
		# embedding vector (and position vector). This is not *strictly* necessary
		# since the [SEP] token unambiguously separates the sequences, but it makes
		# it easier for the model to learn the concept of sequences.
		#
		# For classification tasks, the first vector (corresponding to [CLS]) is
		# used as as the "sentence vector". Note that this only makes sense because
		# the entire model is fine-tuned.
		tokens += [sep_token]
		if sep_token_extra:
			# roberta uses an extra separator b/w pairs of sentences
			tokens += [sep_token]
		segment_ids = [sequence_a_segment_id] * len(tokens)

		if cls_token_at_end:
			tokens += [cls_token]
			segment_ids += [cls_token_segment_id]
		else:
			tokens = [cls_token] + tokens
			segment_ids = [cls_token_segment_id] + segment_ids

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

		# Zero-pad up to the sequence length.
		padding_length = max_seq_length - len(input_ids)
		if pad_on_left:
			input_ids = ([pad_token] * padding_length) + input_ids
			input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
			segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
		else:
			input_ids += ([pad_token] * padding_length)
			input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
			segment_ids += ([pad_token_segment_id] * padding_length)

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		if ex_index < 5:
			logger.info("*** Example ***")
			logger.info("guid: %s", example.guid)
			logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
			logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
			logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
			logger.info("label_id: %s", str(label_id))

		features.append(
				InputFeatures(input_ids=input_ids,
							  input_mask=input_mask,
							  segment_ids=segment_ids,
							  label_id=label_id))
	return features


def get_labels(model_version=2):
	if model_version == 2:
		return [-1, 0, 1, 2]
	else:
		return [-1, 0, 1]


def format_score_sentence_output(bert_input, bert_output):
	"""Format output as list of score\tsentence."""
	new_lines = []
	with open(bert_input) as i, open(bert_output) as o:
		for i_line, o_line in zip(i, o):
			o_line = o_line.strip()
			scores = o_line.split()
			scores = [float(x) for x in scores]
			label = str(np.argmax(scores) - 1)  # Label from -1 to 2.
			i_line = i_line.strip()
			i_line_split = i_line.split('\t')
			s = i_line_split[-1]
			new_line = '\t'.join([label] + [s])
			new_lines.append(new_line)
		return new_lines
