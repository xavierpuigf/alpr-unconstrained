import sys
import ipdb
import cv2
import numpy as np

from glob                                               import glob
from os.path                                    import splitext, basename, isfile
from src.utils                                  import crop_region, image_files_from_folder
from src.drawing_utils                  import draw_label, draw_losangle, write2img
from src.label                                  import lread, Label, readShapes

from pdb import set_trace as pause
import ipdb


YELLOW = (  0,255,255)
RED    = (  0,  0,255)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
#img_files = image_files_from_folder(input_dir)
with open(input_dir, 'r') as f:
    imgs_paths = f.readlines()
    imgs_paths = [x.strip() for x in imgs_paths]
    img_files = [x for x in imgs_paths if len(x) > 0]

for img_file in img_files:

        bname = splitext(basename(img_file))[0]

        I = cv2.imread(img_file)

        detected_cars_labels = '%s/%s_cars.txt' % (output_dir,bname)

        Lcar = lread(detected_cars_labels)

        detected_cars_lp = '%s/%s_lp.txt' % (output_dir,bname)
        cars_lp = readShapes(detected_cars_lp)
        #with open(detected_cars_lp, 'r') as f:
        #    cars_lp = [x.strip() for x in f.readlines()]
        #    cars_lp = Shape() 

        assert len(cars_lp) == len(Lcar)


        sys.stdout.write('%s' % bname)

        if Lcar:

                for i,lcar in enumerate(Lcar):

                        draw_label(I,lcar,color=YELLOW,thickness=3)

                        shape_lp = cars_lp[i]

                        if shape_lp.isValid():

                            pts = shape_lp.pts*lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
                            ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
                            draw_losangle(I,ptspx,RED,3)

        cv2.imwrite('%s/%s_output.png' % (output_dir,bname),I)
        sys.stdout.write('\n')


