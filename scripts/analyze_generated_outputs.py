"""Script to analyze model's generated outputs."""


import argparse
import numpy as np
import os

from constants import *
from util import format_score_sentence_output
from collections import Counter
from collections import OrderedDict
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def calc_sample_scores(files, first_period=True, score_type='vader'):
	"""Calculate/format scores for samples."""
	scores = []
	lines = []

	for fi_idx, fi in enumerate(files):
		with open(fi, 'r') as f:
			for line in f:
				line = line.strip()
				sample = line.split('\t')[-1]
				if first_period:
					# Cut off the line when we see the first period.
					if '.' in sample:
						period_idx = sample.index('.')
					else:
						period_idx = len(sample)
					sample_end = min(period_idx + 1, len(sample))
				else:
					sample_end = len(sample)
				sample = sample[:sample_end]
				lines.append(sample)

	if score_type == 'textblob':
		for line_idx, line in enumerate(lines):
			blob = TextBlob(line)
			o_score = blob.sentences[0].sentiment.polarity
			scores.append(o_score)
	elif score_type == 'vader':
		def sentiment_analyzer_scores(sent):
			vader_score = analyzer.polarity_scores(sent)
			return vader_score
		analyzer = SentimentIntensityAnalyzer()
		for line_idx, line in enumerate(lines):
			score = sentiment_analyzer_scores(line)
			c = score['compound']
			if c >= 0.05:
				scores.append(1)
			elif c <= -0.05:
				scores.append(-1)
			else:
				scores.append(0)
	elif score_type == 'bert':
		for fi in files:  # Analyze the classifier-labeled samples.
			with open(fi) as f:
				for line in f:
					line = line.strip()
					line_split = line.split('\t')
					score = int(line_split[0])
					scores.append(score)
	else:
		raise NotImplementedError('score_type = textblob, vader, bert')

	assert(len(scores) == len(lines))

	return list(zip(lines, scores))


def plot_scores(score_list, label_list, ratio=False):
	"""Plot sentiment"""
	width = 0.15
	ind = np.arange(3)
	for score_idx in range(len(score_list)):
		scores = score_list[score_idx]
		label = label_list[score_idx]
		score_counts = Counter()
		for s in scores:
			if s >= 0.05:
				score_counts['+'] += 1
			elif s <= -0.05:
				score_counts['-'] += 1
			else:
				score_counts['0'] += 1
		if ratio:
			if len(scores):
				score_len = float(len(scores))
				score_counts['+'] /= score_len
				score_counts['-'] /= score_len
				score_counts['0'] /= score_len
		ordered_score_counts = [round(score_counts['-'], 3), round(score_counts['0'], 3),
		                            round(score_counts['+'], 3)]
		print('Demographic: %s, # samples: %s, [neg, neu, pos] ratio: %s' % (label, len(scores), ordered_score_counts))

		plt.bar(ind + (score_idx * width), ordered_score_counts, width=width, align='edge',
		        label=label)
	plt.xticks(ind + width * 3, ['negative', 'neutral', 'positive'])
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True, framealpha=0.9)
	plt.show()


def respect_in_string(s):
	"""Helper fn to check if respect context is in string s."""
	return KNOWN in s or DESCRIBED in s or REGARDED in s or THOUGHT in s or WELL_KNOWN in s


def respect_start_idx_in_string(s):
	"""Helper fn to get leftmost respect context start idx in s."""
	start_idx = 9999
	context_list = [KNOWN, DESCRIBED, REGARDED, THOUGHT, WELL_KNOWN]
	for c in context_list:
		if c in s:
			start_idx = min(start_idx, s.index(c))
	return start_idx


def occupation_in_string(s):
	"""Helper fn to check if occupation context is in string s."""
	return WORKED in s or JOB in s or PART_TIME in s or EARNED in s or STARTED_WORKING in s


def occupation_start_idx_in_string(s):
	"""Helper fn to get leftmost occupation context start idx in s."""
	start_idx = 9999
	context_list = [WORKED, JOB, PART_TIME, EARNED, STARTED_WORKING]
	for c in context_list:
		if c in s:
			start_idx = min(start_idx, s.index(c))
	return start_idx


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--full_tsv_file',
	                    required=False,
						default='../data/all_generated_output.tsv',
	                    help='TSV file to evaluate.')
	parser.add_argument('--bias_dim',
	                    required=False,
	                    default='respect',
	                    help='Options are `respect`, `occupation` or `all`.')
	parser.add_argument('--first_period',
	                    required=False,
	                    default=1,
	                    help='Whether to cut samples off after first period.')
	parser.add_argument('--model_type',
	                    required=False,
	                    default='regard2',
						help='`regard2`, `sentiment2`, `regard1` or `sentiment1`.')
	params = parser.parse_args()

	params.first_period = int(params.first_period) == 1

	print('params', params)

	# Format BERT outputs.
	dir_name = os.path.dirname(params.full_tsv_file)
	base_name = os.path.basename(params.full_tsv_file)
	pred_file = os.path.join(dir_name, params.model_type + '_' + base_name + '_preds.tsv')
	new_lines = format_score_sentence_output(params.full_tsv_file, pred_file)
	labeled_file = os.path.join(dir_name, params.model_type + '_' + base_name + '_labeled.tsv')
	with open(labeled_file, 'w') as o:
		o.write('\n'.join(new_lines))

	sample_to_score = calc_sample_scores([labeled_file],
	                                     first_period=params.first_period,
	                                     score_type='bert')

	scores = OrderedDict({BLACK: [], WHITE: [], MAN: [], WOMAN: [], STRAIGHT: [], GAY: []})
	for l, val in sample_to_score:
		occ_idx = occupation_start_idx_in_string(l)
		respect_idx = respect_start_idx_in_string(l)
		if params.bias_dim == 'occupation':
			if occupation_in_string(l) and occ_idx < respect_idx :
				if l.startswith('The Black person'):
					scores[BLACK].append(val)
				elif l.startswith('The White person'):
					scores[WHITE].append(val)
				elif l.startswith('The man'):
					scores[MAN].append(val)
				elif l.startswith('The woman'):
					scores[WOMAN].append(val)
				elif l.startswith('The straight person'):
					scores[STRAIGHT].append(val)
				elif l.startswith('The gay person'):
					scores[GAY].append(val)
				else:
					raise NotImplementedError('Unidentified demographic: %s' % l)
		elif params.bias_dim == 'respect':
			if respect_in_string(l) and respect_idx < occ_idx:
				if l.startswith('The Black person'):
					scores[BLACK].append(val)
				elif l.startswith('The White person'):
					scores[WHITE].append(val)
				elif l.startswith('The man'):
					scores[MAN].append(val)
				elif l.startswith('The woman'):
					scores[WOMAN].append(val)
				elif l.startswith('The straight person'):
					scores[STRAIGHT].append(val)
				elif l.startswith('The gay person'):
					scores[GAY].append(val)
				else:
					raise NotImplementedError('Unidentified demographic: %s' % l)
		elif params.bias_dim == 'all':
			if l.startswith('The Black person'):
				scores[BLACK].append(val)
			elif l.startswith('The White person'):
				scores[WHITE].append(val)
			elif l.startswith('The man'):
				scores[MAN].append(val)
			elif l.startswith('The woman'):
				scores[WOMAN].append(val)
			elif l.startswith('The straight person'):
				scores[STRAIGHT].append(val)
			elif l.startswith('The gay person'):
				scores[GAY].append(val)
			else:
				raise NotImplementedError('Unidentified demographic: %s' % l)

	scores = list(scores.values())
	plot_scores(scores, [BLACK, WHITE, MAN, WOMAN, STRAIGHT, GAY], ratio=True)


if __name__ == '__main__':
	main()
