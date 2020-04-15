import os
import numpy as np
import cv2
import sys
from typing import *
import toolz

def get_file_extension(file_path: str) -> str:
    return os.path.splitext(file_path)[-1]

def get_file_name(file: str) -> str:
    base_name = os.path.basename(file)
    return os.path.splitext(base_name)[0]

def load_image(path: str) -> np.array:
    return cv2.imread(path)

def rezise_image(img: np.array, dims: Tuple[int,int] = (1000,1000)) -> np.array:
    return cv2.resize(img, dims)

def load_images(path: str, dims: Tuple[int, int] = (1000,1000)) -> Dict:
    files = map(lambda x: os.path.join(path,x) , os.listdir(path))
    files = filter(lambda x: os.path.isfile(x), files)
    files = filter(lambda x: get_file_extension(x) == ".png", files)
    files = dict(map(lambda x: (int(get_file_name(x)), load_image(x)), files))
    files = toolz.valmap(lambda x: rezise_image(x, dims), files)
    return files

def generate_contoure(img: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,canimg = cv2.threshold(gray, 127,255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(canimg, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[0]
    for c in contours:
        if c.shape[0] > contour.shape[0]:
            contour = c
    contour = cv2.approxPolyDP(contour, 10, True)
    contour = contour.reshape((-1,2))
    return contour


def get_cosine(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a-b
    ac = c-b
    ab = ab/np.linalg.norm(ab)
    ac = ac/np.linalg.norm(ac)
    return np.dot(ab,ac)

def get_sine(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    if np.all(a == c) or np.all(a == b):
        return 0
    ab = b-a
    ac = c-a
    ab = ab/np.linalg.norm(ab)
    ac = ac/np.linalg.norm(ac)
    return np.cross(ab,ac)

def is_base(poz, contour, eps):
    """
    Checks if (poz, ..., poz+3) can be base
    """
    c = contour
    n = len(contour)
    first_cos = get_cosine(c[poz], c[(poz+1)%n], c[(poz+2)%n])
    second_cos = get_cosine(c[(poz+1)%n], c[(poz+2)%n], c[(poz+3)%n])
    if abs(first_cos) <= eps and abs(second_cos) <= eps:
        for i in range(3):
            func = lambda x: get_sine(c[(poz+i)%n], c[(poz+i+1)%n],x)
            sines = np.array(list(map(func, contour)))
            if np.sum(sines > 0) > 1:
                return False
        return True

    return False

def get_posible_bases(contour, eps):
    bases = []
    n = len(contour)
    for poz in range(len(contour)):
        if is_base(poz, contour, eps):

            bases.append([(poz+_)%n for _ in range(4)])
    return bases

def find_base(contour):
    n = len(contour)
    angles = map(lambda x: get_cosine(contour[(x-1)%n], contour[x], contour[(x+1)%n]), range(n))

    eps = 0.00
    step = 0.05
    options = []
    while eps < 1.1 and len(options) == 0:
        bases = get_posible_bases(contour, eps)

        if len(bases) > 0:
            res = bases[0]
            func_lenght = lambda ar: sum(map(np.linalg.norm, map(lambda x:x[0]-x[1], zip(ar, ar[1:]))))

            res_lenght = func_lenght(res)
            for b in bases[1:]:
                lenght = func_lenght(b)
                if lenght > res_lenght:
                    res = b
                    res_lenght = lenght

            return res
        eps += step
    return [0,1,2,3] #broken


def generate_description(contour):
    n = len(contour)
    base_idx = find_base(contour)
    base = contour[base_idx]
    top = contour[(np.array(range(n-2))+base_idx[-1])%n]
    return {"base": base, "top": top, "contour": np.array(contour).reshape(-1,2)}


def normalize(description: Dict) -> np.ndarray:
    base_vec = np.array([1,0])
    points = description["top"]

    points = points-points[0]
    cos = np.dot(points[-1], base_vec)/np.linalg.norm(points[-1])
    sin = np.cross(points[-1], base_vec)/np.linalg.norm(points[-1])
    j = np.matrix([[cos, sin], [-sin, cos]])
    points = np.matmul(points, j)
    points = points/sum(np.linalg.norm(x-y) for x,y in zip(points, points[1:]))
    points[:, 0] /= np.max(points[:,0])

    description["top"] = np.array(points).reshape(-1,2)
    description["contour"] = np.array(description["contour"]).reshape(-1,2)

    return description

def add_intermediate_point(contour_dict, step=0.01):
    points = contour_dict["top"]
    res = [points[0]]
    for p in points[1:]:
        direction = p-res[-1]
        norm = np.linalg.norm(direction)
        direction /= norm
        current_dis = step

        while current_dis < norm:
            res.append(res[-1]+direction*step)

            current_dis += step
        res.append(p)
    contour_dict["top_intermediate"] = np.array(res).reshape(-1,2)

    return contour_dict

def load_prepare_data(path):
    data = load_images(path)
    data = toolz.valmap(generate_contoure, data)
    data = toolz.valmap(generate_description, data)
    data = toolz.valmap(normalize, data)
    data = toolz.valmap(add_intermediate_point, data)

    return data


def reverse_points(points):
    points = points.copy()*-1
    points[:,0] += -np.min(points[:,0])

    return points[::-1]


def similarity_angles(points_a, points_b, reverse_a=False, reverse_b=False):
    #points_a = points_a["top_intermediate"]
    #points_b = points_b["top_intermediate"]
    angles_a = np.zeros(3)
    angles_b = np.zeros(3)
    angles_a[1] = 2
    angles_b[1] = 2
   # print(points_a["base"][-2],points_a["base"][-1], points_a["top"][1])
    poz_first = np.where(np.all(points_a["contour"] == points_a["base"][-1], axis=1))[0][0]
    poz_second = np.where(np.all(points_a["contour"] == points_a["base"][-1], axis=1))[0][0]
    n = len(points_a["contour"])
    
    angles_a[0] = get_cosine(points_a["base"][-2],points_a["base"][-1], points_a["contour"][(poz_first+1)%n])
    angles_a[2] = get_cosine(points_a["contour"][(poz_second-2)%n],points_a["contour"][(poz_second-1)%n],points_a["base"][0])
    if len(points_a["top"]) == 3:
        angles_a[1] = get_cosine(points_a["top"][0], points_a["top"][1], points_a["top"][2])
    
    poz_first = np.where(np.all(points_b["contour"] == points_b["base"][-1], axis=1))[0][0]
    poz_second = np.where(np.all(points_b["contour"] == points_b["base"][-1], axis=1))[0][0]
    n = len(points_b["contour"])
    angles_b[0] = get_cosine(points_b["base"][-2],points_b["base"][-1], points_b["contour"][(poz_first+1)%n])
    angles_b[2] = get_cosine(points_b["contour"][(poz_second-2)%n],points_b["contour"][(poz_second-1)%n],points_b["base"][0])
    if len(points_b["top"]) == 3:
        angles_b[1] = get_cosine(points_b["top"][0], points_b["top"][1], points_b["top"][2])
    
    if reverse_a:
        angles_a = angles_a[::-1]
    
    if reverse_b:
        angles_b = angles_b[::-1]
    
    return np.sum((angles_a-angles_b)**2)

def similarity_mse(points_a, points_b, reverse_a=False, reverse_b=False): 
    if points_b["top_intermediate"].shape[0] > points_a["top_intermediate"].shape[0]:
        return similarity_mse(points_b, points_a, reverse_b, reverse_a)
    
    points_a = points_a["top_intermediate"]
    points_b = points_b["top_intermediate"]
   
    if reverse_a:
        points_a = reverse_points(points_a)
    
    if reverse_b:
        points_b = reverse_points(points_b)
   
    f = min(points_a.shape[0], points_b.shape[0])
    if points_a.shape[0] > f:
        dif = points_a.shape[0]-f
        n = np.zeros((f,2))
        a = 0
        p = f/points_a.shape[0]
        res = []
        for i in range(points_a.shape[0]):
            p += f/points_a.shape[0]
            if p >= 1:
                p-= 1
                res.append(points_a[i])
        points_a = np.array(res)
        
            
    
    a = np.ones((points_a.shape[0], 3))
    a[:,:2] = points_a
    b = np.ones((points_b.shape[0], 3))
    b[:,:2] = points_b
    
    t = a.T@a
    t = np.linalg.inv(t)

    t = t@a.T
    t = t@b

    return np.sum((a@t-b)**2)/f
def generate_ranking(data, index):
    data_long = toolz.valfilter(lambda x: len(x["top"]) > 3, data)
    data_short = toolz.valfilter(lambda x: len(x["top"]) <= 3, data)
    
    
    res_short = {}
    res_long = {}
    for k in data_long:
        if k != index:
            res_long[k] = min(similarity_mse(data[index],data_long[k], reverse_b=True),similarity_mse(data_long[k],data[index], reverse_b=True))
 
    for k in data_short:
        if k != index:
            res_short[k] = similarity_angles(data[index],data_short[k], reverse_b=True)+similarity_angles(data_short[k],data[index], reverse_b=True)

    if index in data_short:
        res = sorted(res_short, key=lambda x: res_short[x])+sorted(res_long, key=lambda x: res_long[x])
    else:
        res = sorted(res_long, key=lambda x: res_long[x])+sorted(res_short, key=lambda x: res_short[x])
    return res


if __name__ == "__main__":
   path = sys.argv[1]
   data = load_prepare_data(path)
   for k in sorted(list(data.keys())):
       print(" ".join(map(str, generate_ranking(data, k))))
