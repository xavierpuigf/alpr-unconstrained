import sys, os
from tqdm import tqdm
import keras
import ipdb
import numpy as np
import cv2
import traceback

from src.keras_utils                    import load_model
from glob                                               import glob
from os.path                                    import splitext, basename
from src.utils                                  import im2single, crop_region
from src.keras_utils                    import load_model, detect_lp
from src.label                                  import Shape, writeShapes, lread


def adjust_pts(pts,lroi):
        return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


if __name__ == '__main__':

        try:
                

                input_dir  = sys.argv[1]
                input_file = sys.argv[3]
                output_dir = input_dir

                # ipdb.set_trace()
                with open(input_file, 'r') as f:
                    imgs_paths = f.readlines()
                    imgs_paths = [x.strip() for x in imgs_paths]
                    imgs_paths = [x for x in imgs_paths if len(x) > 0]

                lp_threshold = .5

                wpod_net_path = sys.argv[2]
                wpod_net = load_model(wpod_net_path)

                #imgs_paths = glob('%s/*car.png' % input_dir)

                print('Searching for license plates using WPOD-NET')

                for i,img_path in enumerate(tqdm(imgs_paths)):

                        #print('\t Processing %s' % img_path)

                        bname = splitext(basename(img_path))[0]
                        file_out = '%s/%s_lp.txt' % (output_dir,bname)
                        #if os.path.isfile(file_out):
                        #    continue

                        # Detections
                        det_file = '%s/%s_cars.txt' % (output_dir,bname)
                        print(det_file)
                        if not os.path.isfile(det_file):
                            continue
                        else:
                            cars = lread(det_file)

                        Itotal = cv2.imread(img_path)
                        car_lp = []
                        for id_car, car in enumerate(cars):
                            Ivehicle = crop_region(Itotal, car).astype(np.uint8)


                            ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
                            side  = int(ratio*288.)
                            bound_dim = min(side + (side%(2**4)),608)
                            print( "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

                            Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
                            #ipdb.set_trace()

                            if len(LlpImgs):
                                    #Ilp = LlpImgs[0]
                                    #Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                                    #Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                                    s = Shape(Llp[0].pts)

                                    #cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
                                    if s.isValid():
                                        shape_str = s.tostr() + '_{}'.format(Llp[0].prob())
                                    else:
                                        shape_str = '0'
                            else:
                                shape_str = '0'
                            print(shape_str)
                            #writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
                            car_lp.append(shape_str)
                        with open('%s/%s_lp.txt' % (output_dir,bname), 'w+') as f:
                            #print('%s/%s_lp.txt' % (output_dir,bname))
                            f.write('\n'.join(car_lp))


        except:
                traceback.print_exc()
                sys.exit(1)

        sys.exit(0)


