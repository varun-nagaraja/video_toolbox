import os
import ipdb

import cv
import cv2

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
  '''
  max_frame_over_all = 0
  for i in tracks:
    t = tracks[i].track
    max_frame = max(t.keys())
    if max_frame > max_frame_over_all:
      max_frame_over_all = max_frame

  track_ids_per_frame = [[]] * int(max_frame_over_all+1)
  for i in tracks:
    t = tracks[i]
    for frame in t.track:
      if len(track_ids_per_frame[int(frame)]) == 0:
        track_ids_per_frame[int(frame)] = [t.obj_id]
      else:
        track_ids_per_frame[int(frame)].append(t.obj_id)
      
  video_input = cv2.VideoCapture(input_video_file)
  fps = video_input.get(cv.CV_CAP_PROP_FPS)
  fourcc_code = int(video_input.get(cv.CV_CAP_PROP_FOURCC))
  wd = int(video_input.get(cv.CV_CAP_PROP_FRAME_WIDTH))
  ht = int(video_input.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
  fourcc = cv.CV_FOURCC('X','V','I','D')
  
  video_output = cv2.VideoWriter(output_video_file,fourcc,fps,(wd,ht))
  
  frame_num = 0
  ret, frame_img = video_input.read()
  while ret:
    frame_num += 1
    #frame_img[:] = 0
    if frame_num < len(track_ids_per_frame):
      tracks_in_curr_frame = track_ids_per_frame[frame_num]
      for track in tracks_in_curr_frame:
        bbox = tracks[track].track[frame_num]
        bbox_x1 = int(cap_number(bbox[0],0,wd))
        bbox_y1 = int(cap_number(bbox[1],0,ht))
        if tracks[track].track_format == 'wd_ht':
          bbox_x2 = int(cap_number(bbox[0]+bbox[2],0,wd))
          bbox_y2 = int(cap_number(bbox[1]+bbox[3],0,ht))
        elif tracks[track].track_format == 'two_points':
          bbox_x2 = int(cap_number(bbox[2],0,wd))
          bbox_y2 = int(cap_number(bbox[3],0,ht))
        obj_id = tracks[track].obj_id
        
        box_color = colors[int(obj_id) % len(colors)] 
        cv2.rectangle(frame_img,(bbox_x1,bbox_y1),(bbox_x2,bbox_y2),box_color,10)
        if frame_num in tracks[track].attributes:
          attribute_text = tracks[track].attributes[frame_num]
          if len(attribute_text) > 0: 
            cv2.putText(frame_img,attribute_text,(bbox_x1, bbox_y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,1,box_color,3)
    
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
  

     
