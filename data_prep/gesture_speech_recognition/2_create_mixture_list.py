import os
import numpy as np 
import argparse
import csv
import tqdm
import scipy.io.wavfile as wavfile

np.random.seed(0)

def main(args):
	# read the datalist and separate into train, val and test set
	train_list=[]
	val_list=[]
	test_list=[]

	print("Gathering file names")

	# Get test set list of audios
	for (partition, data_list) in [('test', test_list), ('val', val_list), ('train', train_list)]:
		for path, dirs ,files in os.walk(args.audio_data_direc + partition):
			for filename in files:
				if filename[-4:] =='.wav':
					ln = [path.split('/')[-2], path.split('/')[-1] , filename.split('.')[0]]
					_, audio = wavfile.read(path+'/'+filename)
					ln += [audio.shape[0]/16000]
					data_list.append(ln)
		print(len(data_list))

	# Create mixture list
	print("Creating mixture list")
	f=open(args.mixture_data_list,'w')
	w=csv.writer(f)

	# create test set and validation set
	for (partition, data_list, samples) in [('test', test_list, args.test_samples), ('val', val_list,  args.val_samples), ('train', train_list, args.train_samples)]:
		for _ in range(samples):
			mixtures=[partition]
			cache = []
			shortest = 5000

			idx = np.random.randint(0, len(data_list))
			cache.append(data_list[idx])
			mixtures = mixtures + list(data_list[idx])
			if float(mixtures[-1]) < shortest: shortest = float(mixtures[-1])
			del mixtures[-1]
			mixtures.append(0)

			label = np.random.randint(2)
			if label ==1:
				mixtures = mixtures + list(data_list[idx])
				del mixtures[-1]
				mixtures.append(0)
			else: 
				while len(cache) < args.C:
					idx = np.random.randint(0, len(data_list))
					if data_list[idx] in cache:
						continue
					cache.append(data_list[idx])
					mixtures = mixtures + list(data_list[idx])
					if float(mixtures[-1]) < shortest: shortest = float(mixtures[-1])
					del mixtures[-1]
					mixtures.append(0)
			mixtures.append(shortest)
			mixtures.append(label)
			w.writerow(mixtures)
	f.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='pose dataset')
	parser.add_argument('--data_direc', type=str)
	parser.add_argument('--C', type=int)
	parser.add_argument('--train_samples', type=int)
	parser.add_argument('--val_samples', type=int)
	parser.add_argument('--test_samples', type=int)
	parser.add_argument('--audio_data_direc', type=str)
	parser.add_argument('--mixture_data_list', type=str)
	args = parser.parse_args()
	
	main(args)