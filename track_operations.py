import numpy as np
from scipy.ndimage.filters import gaussian_filter
import functools

class Track:
  def __init__(self, obj_id, obj_type, track_format, operator=None):
    '''
    Parameters
    ----------
    obj_id:          unique id for an object track
    obj_type:        category for the track - person, car etc.
    track_format:    'wd_ht' or 'two_points'
    operator:        a function that takes in a track and frame image and returns a 
                     modified frame. The track variable is fixed to the current
                     track object generating a partial function. 
                     Ex: blur(track, frame_img, frame_index)


    track is a dictionary - {frame: [bbox_x, bbox_y, bbox_wd, bbox_ht], ...}
    attribute is also a dictionary - {frame: 'text', ...}
    '''
    self.obj_id = obj_id
    self.obj_type = obj_type
    self.track = {}
    self.attributes = {}
    self.set_operator(operator)
    if track_format is not 'wd_ht' and track_format is not 'two_points':
      raise ValueError()
    else: 
      self.track_format = track_format 
      
  def set_operator(self, operator):
    if operator is None:
      self._operator = operator
    else:
      new_operator = functools.partial(operator,self)
      self._operator = new_operator
      
  @property
  def operator(self):
    return self._operator
    
  def append_to_track(self,frame,bbox):
    '''
    Appends frame by frame to an existing track.
    
    track = {frame: [bbox_x, bbox_y, bbox_wd, bbox_ht], ...}
    '''
    self.track[int(frame)] = bbox
    
    
  def append_to_attributes(self,attribute):
    '''
    Appends an entire attribute to the existing one.
    
    Attribute is represented as text for each frame.
    '''
    for frame in attribute:
      if frame in self.attributes:
        self.attributes[frame] += "," + attribute[frame]
      else:
        self.attributes[frame] = attribute[frame]
        
  def get_tracklets(self):
    '''
    Returns list of tracklets in the track.
    A track can have breaks and the smaller tracks are called tracklets.
    [(s1,e1),(s2,e2),(s3,e3),...]
    Here start and end frames are inclusive of the track endings.
    '''
    frame_nums = np.hstack((-1, np.array(sorted(self.track.keys()))))
    frame_nums = np.hstack((frame_nums, frame_nums[-1]+2))
    frame_shift = np.array(frame_nums[1:]) - np.array(frame_nums[0:-1])
    frame_jumps = np.where(frame_shift>1)[0]
    tracklets = [(frame_nums[frame_jumps[i]+1], frame_nums[frame_jumps[i+1]]) 
                 for i in range(len(frame_jumps)-1)]
    return tracklets

  
def get_tracklets(trackish):
  '''
  Returns list of tracklets in the tracklike object i.e. a dictionary with
  frames as keys.
  [(s1,e1),(s2,e2),(s3,e3),...]
  Here start and end frames are inclusive of the track endings.
  '''
  frame_nums = np.hstack((-1, np.array(sorted(trackish.keys()))))
  frame_nums = np.hstack((frame_nums, frame_nums[-1]+2))
  frame_shift = np.array(frame_nums[1:]) - np.array(frame_nums[0:-1])
  frame_jumps = np.where(frame_shift>1)[0]
  tracklets = [(frame_nums[frame_jumps[i]+1], frame_nums[frame_jumps[i+1]]) 
               for i in range(len(frame_jumps)-1)]
  return tracklets


def clip_track(track, start_frame, end_frame):
  '''
  Returns a new track object with a track between start_frame and end_frame.
  The frame number starts from 1 and ends at (end_frame - start_frame + 1)
  '''
  clipped_track = Track(track.obj_id, track.obj_type, track.track_format)
  for frame in track.track:
    if frame >= start_frame and frame <= end_frame :
      clipped_track.append_to_track(frame-start_frame+1, track.track[frame])
  
  return clipped_track


def smoothen_track(t):
  '''
  Linear interpolation for missing detections.
  Performs smoothing for the tracklets after interpolation.
  Updates the input track.
  '''
  
  #Interpolation to fill in the gaps of length at most 10 frames
  frame_nums = sorted(t.track.keys())
  for i in range(len(frame_nums)-1):
    curr_frame = frame_nums[i]
    next_frame = frame_nums[i+1]
    diff_frames = next_frame - curr_frame
    if diff_frames < 2 or diff_frames > 10:
      continue
    init_bbox = np.array(t.track[curr_frame])
    final_bbox = np.array(t.track[next_frame])
    diff_bbox = np.divide(np.subtract(final_bbox, init_bbox),float(diff_frames))
    for j in range(1,diff_frames):
      t.track[curr_frame + j] = (init_bbox + j * diff_bbox).tolist() 
    
  # Gaussian smoothing
  # The smoothing occurs only over tracklets with no breaks. Any breaks left over
  # after interpolation will not be touched.
  tracklets = t.get_tracklets()

  for (tracklet_start_frame, tracklet_end_frame) in tracklets:
    bboxes = [t.track[frame] for frame in range(tracklet_start_frame,
                                              tracklet_end_frame+1)]
    smooth_bboxes = _smoothen_tracklet(bboxes).tolist()
    for frame in range(tracklet_start_frame,tracklet_end_frame+1):
      t.track[frame] = smooth_bboxes[frame-tracklet_start_frame]
    
  
def _smoothen_tracklet(bboxes):
  '''
  Applies gaussian filter along the columns of the 2D array bboxes.
  '''
  smooth_bboxes = np.array(bboxes)
  for i in range(smooth_bboxes.shape[1]):
    smooth_bboxes[:,i] = gaussian_filter(smooth_bboxes[:,i],10)
    
  return smooth_bboxes