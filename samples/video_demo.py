import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import ffmpeg
import imageio

import os.path
import time
import argparse
import subprocess

## basic settings
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# make directories if not existed
TEMP_DIR = os.path.join(ROOT_DIR, "samples/temp/")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import coco

## configuration
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def init_weight():
	# Create model object in inference mode.
	model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights trained on MS-COCO
	model.load_weights(COCO_MODEL_PATH, by_name=True)

	print('pretrained model weights loaded')

	return model

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def temp_video_name(filename, number):
	tfilename = filename.split('/')[-1]
	temp = tfilename.split('.')
	temp[-2] = temp[-2] + '_' + str(number)
	tfilename = ".".join(temp)
	return TEMP_DIR + tfilename

def cut_video(filename, start, duration, seg_num=None):
	addon = start if seg_num is None else seg_num

	cut_cmd = ['ffmpeg', '-ss', str(start), '-i', str(filename), '-t', str(duration), \
	 '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', '-b:a', '128k', \
	 temp_video_name(filename, addon)]

	subprocess.call(cut_cmd)
	

def split_video(filename, seg_dur):
	## load video
	probe = ffmpeg.probe(filename)
	video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
	num_frames = int(video_info['nb_frames'])
	fps = int(video_info['avg_frame_rate'].split('/')[0]) / int(video_info['avg_frame_rate'].split('/')[1])
	length = int(num_frames / fps)
	print(video_info)
	print('number of frames:', num_frames, 'fps:', fps, 'video duration in seconds:', length)

	start = 0
	seg_num = 0

	# add a duration to ensure all video contents included
	while start < length + seg_dur:
		cut_video(filename, start, seg_dur, seg_num)
		start += seg_dur
		seg_num += 1

	return seg_num

def merge_video(filename, total_seg):
	# file '/path/to/file1'
	content = ''
	for seg_num in range(int(total_seg)):
		tvout = temp_video_name(filename, str(seg_num)+'__output')
		content = content + "file '%s'\n" % (tvout)
	
	video_outname = temp_video_name(filename, 'finaloutput')
	temp_listname = TEMP_DIR + filename.split('/')[-1].split('.')[-2] + '_list.txt'
	with open(temp_listname, 'w+') as outfile:
		outfile.write(content)

	merge_cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', temp_listname, '-c', 'copy', video_outname]
	print(merge_cmd)
	# subprocess.call(merge_cmd)




def video2np(filename):
	## load video
	probe = ffmpeg.probe(filename)
	video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
	# print('video information:', video_info)
	width = int(video_info['width'])
	height = int(video_info['height'])
	num_frames = int(video_info['nb_frames'])
	

	out, err = (
	    ffmpeg
	    .input(filename)
	    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
	    .run(capture_stdout=True)
	)

	video = (
	    np
	    .frombuffer(out, np.uint8)
	    .reshape([-1, height, width, 3])
	)

	return video

def process_video(infile, color, class_name, duration):

	outfile = temp_video_name(infile, '_output')

	if not color is None:
		color = [float(i) for i in color]
	else:
		# blue as default color
		color = [0.0, 0.0, 1.0]
	print('color:', color)

	if not class_name is None and not class_name in class_names:
		print('class name not found!')
		exit(1)

	if not duration is None and int(duration) >= 1:
		duration = int(duration)

	while os.path.isfile(outfile):
		overwrite = input(outfile + ' exists, overwrite it? (y/n) ')
		if str2bool(overwrite):
			break
		# exit program if do not overwrite
		else:
			outfile = input('enter another filename for output video: ')



	total_seg = split_video(infile, duration)

	# load weight
	model = init_weight()
	start_time = time.time()
	
	for seg_num in range(total_seg):
		# load video to numpy array
		tvin = temp_video_name(infile, seg_num)
		tvout = temp_video_name(tvin, '_output')

		video = video2np(tvin)
		num_frames = video.shape[0]
		print('processing video:', tvin, 'video.shape:', video.shape)

		previous_mask = None
		fps = imageio.get_reader(infile).get_meta_data()['fps']

		start_frame = 0
		seg_num = 0

		## detect and apply mask to each frame in numpy array
		masked_video = np.zeros(video.shape)

		for i in range(video.shape[0]):
			print('frame', i)
			image = video[i]
			# Run detection
			results = model.detect([image]) # verbose=0
			# Visualize results
			r = results[0]
			masked_video[i], previous_mask = visualize.display_bb(image, r['rois'], previous_mask, r['masks'], r['class_ids'], class_names, class_name, color)


		imageio.mimwrite(tvout, masked_video, fps = fps)
		print('saved to', tvout)

			# # concatenate available video segments
			# (
			# 	ffmpeg
			# 	.concat(
			# 		ffmpeg.input(temp_outfile1),
			# 		ffmpeg.input(temp_outfile2),
			# 	)
			# 	.output(temp_outfile3)
			# 	.run()
			# )

			# subprocess.call(['rm', temp_outfile1])
			# subprocess.call(['rm', temp_outfile2])
			# subprocess.call(['mv', temp_outfile3, temp_outfile1])
			# print('concatenated video')


	print('Processing time:', time.time() - start_time, 'Total frames:', num_frames)

	print('total number of segments:', total_seg)
	merge_video(infile, total_seg)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Mask backgroud in video')
	subparser = parser.add_subparsers()

	cut_parser = subparser.add_parser('cut', help = 'cut video')
	cut_parser.add_argument('filename', help='input filename')
	cut_parser.add_argument('start', help='start time')
	cut_parser.add_argument('duration', help='duration')
	cut_parser.set_defaults(func=lambda args: cut_video(args.filename, args.start, args.duration))

	merge_parser = subparser.add_parser('merge', help = 'merge video')
	merge_parser.add_argument('filename', help='input filename')
	merge_parser.add_argument('total_seg', help='total number of video output segmentations')
	merge_parser.set_defaults(func=lambda args: merge_video(args.filename, args.total_seg))

	process_parser = subparser.add_parser('process', help = 'process videos')
	process_parser.add_argument('in_filename', help='Input filename')
	# process_parser.add_argument('out_filename', help='Output filename')
	process_parser.add_argument('--color', nargs='*', help='Enter RGB values (3 float values [0.0, 1.0] in total), or leave it empty as blue')
	process_parser.add_argument('--class_name', nargs='?', help='Name of class to be detected, or leave it empty to detect all classes')
	process_parser.add_argument('--duration', nargs='?', default = 10, help='duration of each video segment in seconds, to avoid cannot allocate memory')
	process_parser.set_defaults(func=lambda args: process_video(args.in_filename, args.color, args.class_name, args.duration))

	args = parser.parse_args()
	args.func(args)
