import os
import time
#dir="/student/zhanghao/model/m2f_50ep_dn_mask_ns0.0_sc1_mode_points_aly_False_no_lb_False_lbns_0.0_pano/"
dir="/student/zhanghao/model/m2f_50ep_dn_mask_ns0.0_sc1_mode_points_aly_False_no_lb_False_lbns_0.0_pano_swin_coco/"
slash='/'
for file in os.listdir(dir):
    if "model_0" in file: # or "model_03" in file:
        print(f"Evaling {file}")
        os.system(f'bash eval_pano_batch_swin.sub {dir+slash+str(file)}')
        time.sleep(2)
