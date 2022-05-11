## Project Structure


`/speaker_extraction`: Scripts to generate 2-speaker mixture for speaker extraction.

`/gesture_speech_recognition`: Scripts to generate list for the GSR network pretraining.


## Dataset File Structure

Prepare the dataset in the following file structure

	pose_ted_long/
	  └── orig/
	    |── train/data.mdb     	# The original train set contains data.mdb & lock.mdb
	    |── val/data.mdb     	# The original val set contains data.mdb & lock.mdb
	    └── test/data.mdb		# The original test set contains data.mdb & lock.mdb

The ZIP file containing data.mdb & lock.mdb can bo downloaded from [here](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context), which is originated from [Youtube Gesture Dataset](https://github.com/youngwoo-yoon/youtube-gesture-dataset).


