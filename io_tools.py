import os
import ipdb
import weakref
import cv
import cv2
import math

colors = [cv.CV_RGB(255,0,0), cv.CV_RGB(0,0,255), cv.CV_RGB(255,255,0), 
          cv.CV_RGB(128,128,128)]


def cap_number(num,low,high):
  '''
  Returns a number that is within [low,high]. 
  If num < low, low is returned.
  if num > high, high is returned.  
  '''
  if num < low:
    return low
  elif num > high:
    return high
  else:
    return num
  

def annotate_frames(input_video_file, output_video_file, tracks):
  '''
  Draws bounding boxes of the tracks along with some text if it exists.
  
  Parameters
  ----------
  tracks            Array of Track objects
  '''
  max_frame_over_all = 0
  for i in range(len(tracks)):
    t = tracks[i].track
    max_frame = max(t.keys())
    if max_frame > max_frame_over_all:
      max_frame_over_all = max_frame

  track_ids_per_frame = [[]] * int(max_frame_over_all+1)
  for i in range(len(tracks)):
    t = tracks[i]
    for frame in t.track:
      if len(track_ids_per_frame[int(frame)]) == 0:
        track_ids_per_frame[int(frame)] = [i]
      else:
        track_ids_per_frame[int(frame)].append(i)
      
  video_input = cv2.VideoCapture(input_video_file)
  fps = video_input.get(cv.CV_CAP_PROP_FPS)
  if math.isnan(fps):
    print "Warning: Setting FPS to 30"
    fps = 30
  fourcc_code = int(video_input.get(cv.CV_CAP_PROP_FOURCC))
  wd = int(video_input.get(cv.CV_CAP_PROP_FRAME_WIDTH))
  ht = int(video_input.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
  fourcc = cv.CV_FOURCC('X','V','I','D')
  
  video_output = cv2.VideoWriter(output_video_file,fourcc,fps,(wd,ht))
  if not video_output.isOpened():
    raise IOError("Could not open output video")
  
  frame_num = 0
  ret, frame_img = video_input.read()
  while ret:
    frame_num += 1
    #frame_img[:] = 0
    if frame_num < len(track_ids_per_frame):
      tracks_in_curr_frame = track_ids_per_frame[frame_num]
      for track in tracks_in_curr_frame:
        if tracks[track].operator is not None:
          frame_img = tracks[track].operator(frame_img, frame_num)  
          
    video_output.write(frame_img)
    ret, frame_img = video_input.read()
    
  video_input.release()
  video_output.release()  


def split_video(in_filename, start_time, end_time, out_filename):
  '''
  Runs a system command to trim the video. Uses ffmpeg.
  '''
  os.system('ffmpeg -i '+in_filename+' -ss '+str(start_time)+
            ' -to '+str(end_time)+ ' -async 1 ' + out_filename)


def box_operator(track, frame_img, frame_num):
  # See here http://stackoverflow.com/a/10791613 for why weakref.
  t = weakref.ref(track)
  bbox = t().track[frame_num]
  (ht, wd, dims) = frame_img.shape
  bbox_x1 = int(cap_number(bbox[0],0,wd-1))
  bbox_y1 = int(cap_number(bbox[1],0,ht-1))
  if t().track_format == 'wd_ht':
    bbox_x2 = int(cap_number(bbox[0]+bbox[2],0,wd-1))
    bbox_y2 = int(cap_number(bbox[1]+bbox[3],0,ht-1))
  elif t().track_format == 'two_points':
    bbox_x2 = int(cap_number(bbox[2],0,wd-1))
    bbox_y2 = int(cap_number(bbox[3],0,ht-1))
  obj_id = t().obj_id
  box_color = colors[int(obj_id) % len(colors)]
  
  cv2.rectangle(frame_img,(bbox_x1,bbox_y1),(bbox_x2,bbox_y2),box_color,10)
  if frame_num in t().attributes:
    attribute_text = str(t().obj_id) + "," + t().attributes[frame_num]
    if len(attribute_text) > 0: 
      cv2.putText(frame_img,attribute_text,(bbox_x1, bbox_y1-5),
                  cv2.FONT_HERSHEY_SIMPLEX,1,box_color,3)
  return frame_img


def blur_operator(track, frame_img, frame_num):
  # See here http://stackoverflow.com/a/10791613 for why weakref.
  t = weakref.ref(track)
  bbox = t().track[frame_num]
  (ht, wd, dims) = frame_img.shape
  bbox_x1 = int(cap_number(bbox[0],0,wd-1))
  bbox_y1 = int(cap_number(bbox[1],0,ht-1))
  if t().track_format == 'wd_ht':
    bbox_x2 = int(cap_number(bbox[0]+bbox[2],0,wd-1))
    bbox_y2 = int(cap_number(bbox[1]+bbox[3],0,ht-1))
  elif t().track_format == 'two_points':
    bbox_x2 = int(cap_number(bbox[2],0,wd-1))
    bbox_y2 = int(cap_number(bbox[3],0,ht-1))
  
  sub_image = frame_img[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
  sub_image_blurred = cv2.blur(sub_image, (10,10))
  frame_img[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = sub_image_blurred
  return frame_img