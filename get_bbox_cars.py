import json
from tqdm import tqdm
import numpy as np
import ipdb

with open('../data/test_automobiles.txt', 'r') as f:
    names = f.readlines()
    names = [x.strip() for x in names]

for name in tqdm(names):
    name_json = name.replace('.jpg', '.json')
    cars = [2927, 3136, 400, 326, 160, 1562, 2879] 
    try:
        with open(name_json, 'r') as f:
            annos = json.load(f)
    except:
        print(name_json)
        continue
    size = np.array(annos['annotation']['imsize'][:2][::-1])
    annos = annos['annotation']['object']
    cars = [anno for anno in annos if anno['name_ndx']-1 in cars]
    line_str = []
    for car in cars:
        poly = car['polygon']
        xmin = min(poly['x'])
        xmax = max(poly['x'])
        ymin = min(poly['y'])
        ymax = max(poly['y'])
        pmin = np.array([xmin, ymin])
        pmax = np.array([xmax, ymax])
        center = (pmin + pmax)/(2.*size)
        size_norm = (pmax - pmin)/(size)
        str_car = '0 {} {} {} {}'.format(center[0], center[1], size_norm[0], size_norm[1])
        line_str.append(str_car)

    name_txt = 'output_annos/{}_cars.txt'.format(name.split('/')[-1].split('.')[0])
    with open(name_txt, 'w+') as f:
        f.writelines([x+'\n' for x in line_str])
    
    #ipdb.set_trace()

