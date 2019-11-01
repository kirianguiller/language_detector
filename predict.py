import time, glob, os
import numpy as np
from ngram import Ngram, typical
from collections import Counter



def read_in_chunks(file_object, chunk_size=1024):
	"""Lazy function (generator) to read a file piece by piece.
	Default chunk size: 1k.
    chunk_size will be approximately equal to the number of characters, and not equal to the number of lines
    """

	while True:
		data = file_object.readlines(chunk_size)
		if not data:
			break
		yield data



# ngram = Ngram(make_new = True, ngramnum = 500)
ngram = Ngram(make_new = False, ngramnum = 500)

# chunk_size will be approximately equal to the number of characters, and not equal to the number of lines
chunk_size = 500000

data_folder="data/"
input_folder = data_folder + 'input/'
output_folder = data_folder + 'output/'



for input_file in glob.glob(input_folder+'*'):
	if os.path.isfile(input_file):
		file_name = input_file.split('/')[-1]
		output_file = output_folder + file_name + '.pred'
		list_pred_file = []
		number_of_lines = 0
		number_of_chars = 0
		print('---------- File : {}  -----------'.format(input_file))


		with open(input_file, 'r+',encoding='utf-8') as f_in, open(output_file, 'w+',encoding='utf-8') as f_out:
			ti = time.time()

			for chunk in read_in_chunks(f_in, chunk_size = chunk_size):

				languages_list, score_list = ngram.chunck_predict(chunk)
				list_pred_file += languages_list
				number_of_lines += len(chunk)

				for lang, line, score in zip(languages_list, chunk, score_list):
					f_out.write(lang+ ' ({}) '.format(score) + ':::' + line)


			tf = time.time()
			print('total time :', tf - ti, "seconds")
			print('time per line', (tf - ti)/number_of_lines, 'seconds')
			# print('char per line :',number_of_chars/number_of_lines)
			print('number of lines :', number_of_lines)
			print('Distribution :', Counter(list_pred_file))
			print()




