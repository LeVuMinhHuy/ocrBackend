# -*- coding: utf-8 -*-

import subprocess
import os
# from google.colab import drive
# drive.mount('/content/drive')
# # %cd "/content/drive/My Drive/ocr_demo_code"
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT
#OrderedDict: dictionary subclass that remembers the order that keys were first inserted
from collections import OrderedDict
import copy
import math
# !git clone https://github.com/mauvilsa/imgtxtenh
# !dpkg --configure -a
# !apt-get install libmagickwand-dev
# # %cd  ./imgtxtenh
# !cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=Release
# !make

def preprocessing(img_path): 
  # example: img_path='/content/drive/My Drive/ocr_demo_code/test_imgs/test_1708/0.jpg'
  
  pre_img_path = 'pre_' + os.path.basename(img_path) # pre_0.jpg
  command = './imgtxtenh/imgtxtenh ' + os.path.basename(img_path) + ' -p ' + os.path.basename(pre_img_path)# ../../imgtxtenh/imgtxtenh 0.jpg -p pre_0.jpg
  print("aa", command)
  p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
  return pre_img_path

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    # print("X: ",x)
  
    if cuda:
        x = x.cuda()


    #forward pass
    with torch.no_grad():
      y, feature = net(x)
      print(y, feature)

    # make score and link map
    
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    # print("Score_text: ", score_text)
    # print("Score_link: ", score_link)


    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()
    t0 = time.time() - t0
    t1 = time.time()


    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    t1 = time.time() - t1


    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    # print("Render image: ", render_img)
    # plt.imshow(ret_score_text)
    # print("Bounding Box: ", polys)

    # if show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

class Point:
    '''
    Each point have 2 main values: coordinate(lat, long) and cluster_id
    '''
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
        self.cluster_id = UNCLASSIFIED

    def __repr__(self):
        return '(x:{}, y:{}, id:{}, cluster:{})' \
            .format(self.x, self.y, self.id, self.cluster_id)

#In G-DBScan we use elip instead of circle to cluster (because we mainly use for horizontal text image --> elip is more useful)
def n_pred(p1, p2):
#     return (p1.x - p2.x)**2/160000 + (p1.y - p2.y)**2/2500 <= 1
#     print(p1.x -p2.x)
#     print(p1.y -p2.y)
#     return (p1.x - p2.x)**2/50000 + (p1.y - p2.y)**2/1500 <= 1
#     return (p1.x - p2.x)**2/20000 + (p1.y - p2.y)**2/1300 <= 1
#     return (p1.x - p2.x)**2/2000 + (p1.y - p2.y)**2/130 <= 1
      return (p1.x - p2.x)**2/500 + (p1.y - p2.y)**2/70 <= 1
#     return (p1.x - p2.x)**2/3500 + (p1.y - p2.y)**2/150 <= 1
#     return (p1.x - p2.x)**2/7000 + (p1.y - p2.y)**2/1300 <= 1
#     return (p1.x - p2.x)**2/8000 + (p1.y - p2.y)**2/300 <= 1
#     return (p1.x - p2.x)**2/17000 + (p1.y - p2.y)**2/300 <= 1
#     return (p1.x - p2.x)**2/13000 + (p1.y - p2.y)**2/250 <= 1
#    return (p1.x - p2.x)**2/15000 + (p1.y - p2.y)**2/180 <= 1


def w_card(points):
    return len(points)

UNCLASSIFIED = -2
NOISE = -1

def GDBSCAN(points, n_pred, min_card, w_card):
    points = copy.deepcopy(points)
    cluster_id = 0
    for point in points:
        if point.cluster_id == UNCLASSIFIED:
            if _expand_cluster(points, point, cluster_id, n_pred, min_card,
                               w_card):
                cluster_id = cluster_id + 1
    clusters = {}
    for point in points:
        key = point.cluster_id
        if key in clusters:
            clusters[key].append(point)
        else:
            clusters[key] = [point]
    return list(clusters.values())


def _expand_cluster(points, point, cluster_id, n_pred, min_card, w_card):
    if not _in_selection(w_card, point):
        points.change_cluster_id(point, UNCLASSIFIED)
        return False

    seeds = points.neighborhood(point, n_pred)
    if not _core_point(w_card, min_card, seeds):
        points.change_cluster_id(point, NOISE)
        return False

    points.change_cluster_ids(seeds, cluster_id)
    seeds.remove(point)

    while len(seeds) > 0:
        current_point = seeds[0]
        result = points.neighborhood(current_point, n_pred)
        if w_card(result) >= min_card:
            for p in result:
                if w_card([p]) > 0 and p.cluster_id in [UNCLASSIFIED, NOISE]:
                    if p.cluster_id == UNCLASSIFIED:
                        seeds.append(p)
                    points.change_cluster_id(p, cluster_id)
        seeds.remove(current_point)
    return True


def _in_selection(w_card, point):
    return w_card([point]) > 0


def _core_point(w_card, min_card, points):
    return w_card(points) >= min_card


class Points:
    'Contain list of Point'
    def __init__(self, points):
        self.points = points

    def __iter__(self):
        for point in self.points:
            yield point

    def __repr__(self):
        return str(self.points)

    def get(self, index):
        return self.points[index]

    def neighborhood(self, point, n_pred):
        return list(filter(lambda x: n_pred(point, x), self.points))

    def change_cluster_ids(self, points, value):
        for point in points:
            self.change_cluster_id(point, value)

    def change_cluster_id(self, point, value):
        index = (self.points).index(point)
        self.points[index].cluster_id = value

    def labels(self):
        return set(map(lambda x: x.cluster_id, self.points))

trained_model_path = './craft_mlt_25k.pth'
net = CRAFT()
net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location='cpu')))
net.eval()

# !sudo apt install tesseract-ocr
# !pip install pytesseract
import pytesseract

def img_to_text(image):
  # Initialize CRAFT parameters
  text_threshold = 0.7
  low_text = 0.4
  link_threshold =0.4
  # cuda = True
  cuda=False
  canvas_size =1280
  mag_ratio =1.5
  #if text image present curve --> poly=true
  poly=False
  refine=False
  show_time=False
  refine_net = None
  
  print("1,1")
  bboxes, polys, score_text = test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net)

  # file_utils.saveResult(img_path, image[:,:,::-1], bboxes, dirname='/content/drive/My Drive/ocr_demo_code/craft_result/')

  poly_indexes = {}
  central_poly_indexes = []
  for i in range(len(polys)):
    poly_indexes[i] =  polys[i]
    x_central = (polys[i][0][0] + polys[i][1][0] +polys[i][2][0] + polys[i][3][0])/4
    y_central = (polys[i][0][1] + polys[i][1][1] +polys[i][2][1] + polys[i][3][1])/4
    central_poly_indexes.append({i: [int(x_central), int(y_central)]})

  X = []
  for idx, x in enumerate(central_poly_indexes):
    point = Point(x[idx][0],x[idx][1], idx)
    X.append(point)

  
  

  poly=False
  refine=False
  show_time=False
  refine_net = None
  
  clustered = GDBSCAN(Points(X), n_pred, 1, w_card)

  cluster_values = []
  for cluster in clustered:
    sort_cluster = sorted(cluster, key = lambda elem: (elem.x, elem.y))
    max_point_id = sort_cluster[len(sort_cluster) - 1].id
    min_point_id = sort_cluster[0].id
    max_rectangle = sorted(poly_indexes[max_point_id], key = lambda elem: (elem[0], elem[1]))
    min_rectangle = sorted(poly_indexes[min_point_id], key = lambda elem: (elem[0], elem[1]))

    right_above_max_vertex = max_rectangle[len(max_rectangle) -1]
    right_below_max_vertex = max_rectangle[len(max_rectangle) -2]
    left_above_min_vertex = min_rectangle[0] 
    left_below_min_vertex = min_rectangle[1]
    
    if (int(min_rectangle[0][1]) > int(min_rectangle[1][1])): 
        left_above_min_vertex = min_rectangle[1]
        left_below_min_vertex =  min_rectangle[0]
    if (int(max_rectangle[len(max_rectangle) -1][1]) < int(max_rectangle[len(max_rectangle) -2][1])):
        right_above_max_vertex = max_rectangle[len(max_rectangle) -2]
        right_below_max_vertex = max_rectangle[len(max_rectangle) -1]
            
    cluster_values.append([left_above_min_vertex, left_below_min_vertex, right_above_max_vertex, right_below_max_vertex])

  # file_utils.saveResult(image_path, image[:,:,::-1], cluster_values, dirname='/content/drive/My Drive/ocr_demo_code/cluster_result/')

  img = np.array(image[:,:,::-1])

  res = []
  for i, box in enumerate(cluster_values):
    poly = np.array(box).astype(np.int32).reshape((-1))
    poly = poly.reshape(-1, 2)

    rect = cv2.boundingRect(poly)
    x,y,w,h = rect
    # croped = img[y:y+h, x:x+w].copy()
    croped = img[y:y+h, x:x+w].copy()
  
    # Preprocess croped segment
    croped = cv2.resize(croped, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
    croped = cv2.cvtColor(croped, cv2.COLOR_BGR2GRAY)
    croped = cv2.GaussianBlur(croped, (3, 3), 0)
    croped = cv2.bilateralFilter(croped,5,25,25)
    croped = cv2.dilate(croped, None, iterations=1)
    croped = cv2.threshold(croped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # croped = cv2.threshold(croped, 90, 255, cv2.THRESH_BINARY)[1]
    # croped = cv2.cvtColor(croped, cv2.COLOR_BGR2RGB)

    text = pytesseract.image_to_string(croped, lang='eng')
    res.append(text)
  return res

def X_Y_localization(listText):
  new_text = []
  res1 = []
  res2 = []
  count = {}
  for text in listText:
    # obmit space and endl in text
    text_without_space_endl = ''
    for word in text.split():
      text_without_space_endl += word
    
    if len(text_without_space_endl) < 4 or len(text_without_space_endl) > 11:
      continue
    # modify some easy to misunderstanding features in the text
    norm_text = ''
    for i in range(len(text_without_space_endl)):
        if text_without_space_endl[i] in ['S','$', '£']:
          norm_text += '5'
        elif text_without_space_endl[i] in ['I','§', 'V', 'a']:
          norm_text += '1'
        else:
          norm_text += text_without_space_endl[i]
    new_text.append(norm_text)
    # Check the frequency of first 4 letters in norm_text
    first_2 = norm_text[0:2]  
    if first_2 in count and first_2[1].isdigit():
      count[first_2] += 1
    else:
      count[first_2] = 1  
  
  max = 0
  str1 = ''
  for i in count:
    if count[i] > max:
      max = count[i]
      str1 = i
  max = 0
  str2 = ''
  for i in count:
    if i != str1 and count[i] > max:
      max = count[i]
      str2 = i
  if not str1.isdigit() or not str2.isdigit():  # 4 so dau tien cuc ki quan trong, neu k xac dinh chinh xac 1 trong 2 thi xem nhu khong tim dc
    return res1, res2   # res1 = res2 = [] 

  avg_len1 = 0
  count1 = 0
  avg_len2 = 0
  count2 = 0
  for text in new_text:
    if text[0:2] == str1:
      res1.append(text)
      for dot in ['.',',']:
        if dot in text and text.index(dot) == 6:
          avg_len1 += 6
          count1 += 1
        elif dot in text and text.index(dot) == 7:
          avg_len1 += 7
          count1 += 1
    elif text[0:2] == str2:
      res2.append(text)
      for dot in ['.',',']:
        if dot in text and text.index(dot) == 6:
          avg_len2 += 6
          count2 += 1
        elif dot in text and text.index(dot) == 7:
          avg_len2 += 7
          count2 += 1

  if count1 == 0:
    avg_len1 = 0
  else:
    avg_len1 = avg_len1 / count1
  if count2 == 0:
    avg_len2 = 0
  else:
    avg_len2 = avg_len2 / count2
  
  if avg_len1 > avg_len2:
    return res1, res2
  else:
    return res2, res1

def norm_X_Y(listX, listY):
  if listX == [] or listY == []:
    # print('Cannot read X,Y from img')
    return [],[]
  float_listX = []
  float_listY = []
  for text in listX:
    if len(text) < 7:
      continue
    norm_text = ''
    if text.count('.') > 1 or ('.' in text and text.index('.') != 7):
      # neu xhien >1 '.' or dau '.' nam sai vi tri (sau chu so thu 7) thi bo toan bo '.' roi them lai sau
      for letter in text:
        if letter != '.':
          norm_text += letter
      text = norm_text
      norm_text = ''
    for i in range(len(text)):
      if text[i] == ',':
        norm_text += '.'
      elif text[i] != '.' and not text[i].isdigit():
        norm_text += '5'  # Vi cac so sau sai lech khong anh huong qua lon den toa do nen chuan hoa cac ki tu k phai so thanh 5
      else:
        norm_text += text[i]
    # Xu ly cac truong hop khong co '.'
    if norm_text.count('.') == 0:
      text = ''
      for i in range(len(norm_text)):
        if i == 6:
          text += norm_text[i] + '.'
        else:
          text += norm_text[i]
    else:
      text = norm_text
    float_listX.append(float(text))
  
  for text in listY:
    if len(text) < 6:
      continue
    norm_text = ''
    if text.count('.') > 1 or ('.' in text and text.index('.') != 6): 
      # neu xhien >1 '.' or dau '.' nam sai vi tri (sau chu so thu 6) thi bo toan bo '.' roi them lai sau
      for letter in text:
        if letter != '.':
          norm_text += letter
      text = norm_text
      norm_text = ''
    for i in range(len(text)):
      if text[i] == ',':
        norm_text += '.'
      elif text[i] != '.' and not text[i].isdigit():
        norm_text += '5'  # Vi cac so sau sai lech khong anh huong qua lon den toa do nen chuan hoa cac ki tu k phai so thanh 5
      else:
        norm_text += text[i]
    # Xu ly cac truong hop khong co '.'
    if norm_text.count('.') == 0:
      text = ''
      for i in range(len(norm_text)):
        if i == 5:
          text += norm_text[i] + '.'
        else:
          text += norm_text[i]
    else:
      text = norm_text
    float_listY.append(float(text))
  return float_listX, float_listY

def get_X_Y (X,Y):
  res_x = 0
  res_y = 0
  if X == [] or Y == []:
    # print('Cannot get X,Y. Try another img')
    return res_x, res_y
  found_x = 0
  found_y = 0
  for x in X:
    for another_x in X:
      if abs(another_x - x) > 0 and abs(another_x - x) < 10:
        found_x += 1
        res_x = (another_x + x) / 2
        break
    if found_x > 0:
      break

  for y in Y:
    for another_y in Y:
      if abs(another_y - y) > 0 and abs(another_y - y) < 10:
        found_y += 1
        res_y = (another_y + y) / 2
        break
    if found_y > 0:
      break
    
  if found_x == 0: # lay trung binh cong
    for x in X:
      res_x += x
    res_x = res_x / len(X)
  if found_y == 0: # lay trung binh cong
    for y in Y:
      res_y += y
    res_y = res_y / len(Y)

  return res_x, res_y

# !pip install pyproj
import pyproj
import psycopg2


def vn2k_to_wgs83(coordinate,crs): 
    """
    Đây là hàm chuyển đổi cặp toạ độ x, y theo vn2k sang kinh độ , vĩ độ theo khung toạ độ của Google Map 
    Công thức này được cung cấp bởi thư viện pyproj 
    
    Input:
    
        - ( x, y ) : TUPLE chứa cặp toạ độ x và y theo đơn vị float 
        - crs : INT - id (mã) vùng chứa cặp toạ độ x, y theo toạ độ Google
    Output: 
        - (longitude, latitude): TUPLE chứa cặp kinh độ - vĩ độ theo toạ độ Google Map
    """
    new_coordinate = pyproj.Transformer.from_crs(
        crs_from=crs, crs_to=4326, always_xy=True).transform(coordinate[1], coordinate[0])

    return new_coordinate[1], new_coordinate[0]

def process(img_path, city_id):
  print("1")
  pre_img_path = preprocessing(img_path)
  # from IPython.display import Image 
  # Image(pre_img_path)
  image = imgproc.loadImage(pre_img_path)
  if city_id == 1: # for example, TP.HCM
    crs = 9210
  listText = img_to_text(image)
  # print(listText)
  print("2")

  listX, listY = X_Y_localization(listText)

  X,Y = norm_X_Y(listX,listY)
  # print(X,Y)


  x,y = get_X_Y(X,Y)
  print(x,y)



  if x != 0 or y != 0:
    lat, lng = vn2k_to_wgs83((x, y),crs)
    # print(lat, lng)
    # from geopy.geocoders import Nominatim
    # geolocator = Nominatim(user_agent="AIzaSyABrrKL3I5HZh7wx9QLCk7H5Rq0TrdtLjw")
    # location = geolocator.reverse(str(lat) + ', ' + str(lng))
    # print(location.address)
    print(lat,lng)
    return lat, lng
  else:
    return 0,0

if __name__ == "__main__":
  process('./newimage.jpg',1)
