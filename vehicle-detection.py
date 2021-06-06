import sys
import os
from tqdm import tqdm
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from src.label                          import Label, lwrite
from os.path                            import splitext, basename, isdir
from os                                         import makedirs
from src.utils                          import crop_region, image_files_from_folder
from darknet.python.darknet import detect


if __name__ == '__main__':

        try:
        
                input_dir  = sys.argv[1]
                output_dir = sys.argv[2]

                vehicle_threshold = .5

                vehicle_weights = b'data/vehicle-detector/yolo-voc.weights'
                vehicle_netcfg  = b'data/vehicle-detector/yolo-voc.cfg'
                vehicle_dataset = b'data/vehicle-detector/voc.data'
                print("Loading...")

                vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
                vehicle_meta = dn.load_meta(vehicle_dataset)
                
                with open(input_dir, 'r') as f:
                    imgs_paths = f.readlines()
                    imgs_paths = [x.strip() for x in imgs_paths]
                    imgs_paths = [x for x in imgs_paths if len(x) > 0]
                print(len(imgs_paths))
                print(imgs_paths)
                imgs_paths.sort()

                if not isdir(output_dir):
                        makedirs(output_dir)

                print('Searching for vehicles using YOLO...')

                for i,img_path in enumerate(tqdm(imgs_paths)):

                        #print('\tScanning %s' % img_path)

                        bname = basename(splitext(img_path)[0])
                        file_out = '%s/%s_cars.txt' % (output_dir,bname)
                        if os.path.isfile(file_out):
                            continue

                        R,_ = detect(vehicle_net, vehicle_meta, img_path.encode('utf-8') ,thresh=vehicle_threshold)

                        R = [r for r in R if r[0] in ['car','bus']]

                        print('\t\t%d cars found' % len(R))

                        if len(R):

                                Iorig = cv2.imread(img_path)
                                WH = np.array(Iorig.shape[1::-1],dtype=float)
                                Lcars = []

                                for i,r in enumerate(R):

                                        cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
                                        tl = np.array([cx - w/2., cy - h/2.])
                                        br = np.array([cx + w/2., cy + h/2.])
                                        label = Label(0,tl,br)
                                        Icar = crop_region(Iorig,label)

                                        Lcars.append(label)

                                        #cv2.imwrite('%s/%s_%dcar.png' % (output_dir,bname,i),Icar)

                                lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)

        except:
                traceback.print_exc()
                sys.exit(1)

        sys.exit(0)
        
