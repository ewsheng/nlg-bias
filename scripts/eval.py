"""Script to evaluate model outputs."""


import argparse
import os
import subprocess


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--sample_file',
	                    required=False,
	                    default='data/generated_samples/small_gpt2_generated_samples.tsv',
	                    help='Sample file to label with classifier and evaluate.')
	parser.add_argument('--model_type',
	                    required=False,
	                    default='regard2',
	                    help='`regard2`, `sentiment2`, `regard1` or `sentiment1`.')

	params = parser.parse_args()

	print('params', params)

	# Use classifier to label samples.
	data_file = params.sample_file + '.XYZ'
	no_ext_sample_name = params.sample_file.split('.')[0]  # Without file extension.
	if not os.path.exists(data_file + '_preds'):
		run_classifier = 'bash scripts/run_ensemble.sh ' + params.model_type + ' ' + no_ext_sample_name
		p = subprocess.Popen(run_classifier, shell=True)
		p.communicate()

	# Calculate ratios of pos/neu/neg samples for evaluation.
	print('=' * 80)
	print('RESPECT')
	p = subprocess.Popen('python scripts/analyze_generated_outputs.py --full_tsv_file ' + params.sample_file +
	                     ' --bias_dim "respect" --model_type ' + params.model_type, shell=True)
	p.communicate()
	print('=' * 80)
	print('OCCUPATION')
	p = subprocess.Popen('python scripts/analyze_generated_outputs.py --full_tsv_file ' + params.sample_file +
	                     ' --bias_dim "occupation" --model_type ' + params.model_type, shell=True)
	p.communicate()


if __name__ == '__main__':
	main()
