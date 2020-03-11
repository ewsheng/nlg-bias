"""Simple majority ensemble of different runs of regard/sentiment classifiers."""


import argparse
import collections
import os
import numpy as np

from sklearn.metrics import accuracy_score


def eval_majority_ensemble(args):
	"""Evaluate majority ensemble predictions."""
	files = os.listdir(args.data_dir)
	sample_to_labels = collections.OrderedDict()
	samples = []
	for fi in files:
		with open(os.path.join(args.data_dir, fi), 'r') as f:
			for line_idx, line in enumerate(f):
				line = line.strip()
				split = line.split('\t')
				label = int(split[0])
				sample = split[1]
				if line_idx not in sample_to_labels:  # Iterating through first file.
					sample_to_labels[line_idx] = collections.Counter()
					samples.append(sample)
				sample_to_labels[line_idx].update([label])

	# Keep majority label.
	majority_list = []
	majority_labels = []
	for sample, labels in zip(samples, sample_to_labels.values()):
		label, label_count = labels.most_common(1)[0]  # Choose first value if no agreement.
		label_count = [str(labels[x]) for x in range(-1, 3)]
		majority_list.append('\t'.join(label_count))
		majority_labels.append(label)

	if args.groundtruth_file:
		# Read groundtruth file.
		groundtruth_labels = []
		with open(args.groundtruth_file, 'r') as f:
			for line in f:
				line = line.strip()
				label = int(line.split('\t')[0])
				groundtruth_labels.append(label)

		# Evaluate accuracy.
		print('Accuracy:', accuracy_score(groundtruth_labels, majority_labels))

	# Output count per label to file.
	pred_output_file = args.output_prefix + '_preds.tsv'
	with open(pred_output_file, 'w') as o:
		o.write('\n'.join(majority_list))


def reveal_demographics(args):
	"""Remove demographic masks and replace original demographics in samples."""
	samples = []
	with open(args.file_with_demographics, 'r') as f:
		for line in f:
			line = line.strip()
			sample = line.split('\t')[-1]
			samples.append(sample)
	labels = []
	pred_output_file = args.output_prefix + '_preds.tsv'
	with open(pred_output_file, 'r') as f:
		for line in f:
			line = line.strip()
			label_set = [float(x) for x in line.split('\t')]
			label = np.argmax(label_set) - 1
			labels.append(str(label))
	unmasked_output_file = args.output_prefix + '_labeled.tsv'
	with open(unmasked_output_file, 'w') as f:
		for sample, label in zip(samples, labels):
			f.write('\t'.join([label, sample]) + '\n')
		f.write('\n')


def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument(
		'--data_dir',
		default='',
		type=str,
		required=True,
		help='data_dir should contain prediction files for the same inputs. Files are in the format label\tsample.',
	)
	parser.add_argument(
		'--file_with_demographics',
		default='',
		type=str,
		required=True,
		help='File of actual samples (without masked demographics) to correspond to labels.'
	)
	parser.add_argument(
		'--output_prefix',
		default='',
		type=str,
		required=True,
		help='Prefix of files to output ensemble results.'
	)
	parser.add_argument(
		'--groundtruth_file',
		default='',
		type=str,
		required=False,
		help='File with groundtruth labels, only used for calculating scores. File is in the format label\tsample.'
	)

	args = parser.parse_args()

	eval_majority_ensemble(args)

	reveal_demographics(args)


if __name__ == '__main__':
	main()
