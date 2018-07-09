# Mask background in video (for python3, tested in Ubuntu 16.04 LTS)

## Additional requirements
1. ffmpeg-python
2. imageio

## How to run
1. Download pre-trained COCO weights to `path/to/Mask_RCNN/` (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
2. cd to `path/to/Mask_RCNN/samples/` and move the input mp4 videos here 
3. `python3 video_demo.py [INPUT FILENAME] [OUTPUT FILENAME] [optional COLOR] [optional CLASS NAME]` (run `python3 video_demo.py -h` to see more details)
4. An example: `python3 video_demo.py input.mp4 output.mp4 --color 0.0 1.0 0.0 --class_name cat` (background color set to green)
   available class names from COCO
```
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
```

## Trime video under Ubuntu
1. `ffmpeg -i input.mp4 -ss 00:00:03 -t 00:00:08 -async 1 output.mp4` (-ss start time, -t duration)
2. `ffmpeg -i input.mp4 -ss 00:00:00 -t 00:00:05 -c copy output.mp4`

## Run speed
Testing Hardware: i5-8400, 16GB RAM, GTX1060(3GB)  


Video resolution  | Video length          | Processing time  | Processing speed (fps)  
----------------- | --------------------- | ---------------- | ---------------------- 
1280x720          | 6s, 151 frames        | 101.32s          | 1.49 
1280x720          | 10s, 291 frames       | 167.86s          | 1.73 
1280x720          | 10s, 300 frames       | 195.69s          | 1.53 
1280x720          | 15s, 449 frames       | 247.95s          | 1.81
