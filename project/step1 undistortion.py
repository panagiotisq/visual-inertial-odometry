import numpy as np
import cv2
import os
import re

##απο το yalm αρχειο
distortion_coeffs_cam0= np.array([0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202,0.00020293673591811182])
distortion_coeffs_cam1=np.array([0.0034003170790442797, 0.001766278153469831, -0.00266312569781606,0.0003299517423931039])
##intrinsics: [focal length x, focal length y, center in pixel x, center in pixel y]
intrinsics_cam0= np.array([190.97847715128717, 190.9733070521226, 254.93170605935475, 256.8974428996504])
intrinsics_cam1=np.array([190.44236969414825, 190.4344384721956, 252.59949716835982, 254.91723064636983])

#Homogenous transformation from imu to cameras
T_cam0=np.array([[-0.9995250378696743, 0.029615343885863205, -0.008522328211654736, 0.04727988224914392]
    ,[0.0075019185074052044, -0.03439736061393144, -0.9993800792498829, -0.047443232143367084]
    ,[-0.02989013031643309, -0.998969345370175, 0.03415885127385616, -0.0681999605066297]
    ,[0.0, 0.0, 0.0, 1.0]])
T_cam1=np.array([[-0.9995110484978581, 0.030299116376600627, -0.0077218830287333565, -0.053697434688869734]
    ,[0.008104079263822521, 0.012511643720192351, -0.9998888851620987, -0.046131737923635924]
    ,[-0.030199136245891378, -0.9994625667418545, -0.012751072573940885, -0.07149261284195751]
    ,[0.0, 0.0, 0.0, 1.0]])

##Homogennous transformation from cam1 to cam0
Tc1c0=np.array([[0.9999994457734953, -0.0008233639921576076, -0.0006561436136444361, -0.10106110275180535]
    ,[0.0007916877528168117, 0.9988994619156757, -0.04689603624058988, -0.0019764575873431013]
    ,[0.0006940340102242531, 0.04689549078870055, 0.9988995601463054, -0.0011756424802046581]
    ,[0.0, 0.0, 0.0, 1.0]])

##Στο equidistant model, οι συντελεστές αντιστοιχούν σε ακτινική παραμόρφωση της μορφής: rd=r*(1+k1r**2+k2r**4+k3r**6+k4r**8+...)
##με r αποσταση απο κεντρικο σημειο
K0 = np.array([[intrinsics_cam0[0],0,intrinsics_cam0[2]],[0,intrinsics_cam0[1],intrinsics_cam0[3]],[0,0,1]])
D0 = distortion_coeffs_cam0
##τοποθεσιες δεδομενων εισοδου
input_dir = "dataset-corridor4_512_16\mav0\cam0\data"
output_dir = "cam0_images_undistorted/"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, img_name))
    #λυνει επαναληπτικα για να βρει τα πραγματικα x,y 
    undistorted_img = cv2.fisheye.undistortImage(img, K0, D0, Knew=K0 )
    cv2.imwrite(os.path.join(output_dir, img_name), undistorted_img)

##cam1 undistortion (ιδια διαδικασια)
K1 = np.array([[intrinsics_cam1[0],0,intrinsics_cam1[2]],[0,intrinsics_cam1[1],intrinsics_cam1[3]],[0,0,1]])
D1 = distortion_coeffs_cam1

input_dir = "dataset-corridor4_512_16\mav0\cam1\data"
output_dir = "cam1_images_undistorted/"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, img_name))
    undistorted_img = cv2.fisheye.undistortImage(img, K1, D1, Knew=K1 )
    cv2.imwrite(os.path.join(output_dir, img_name), undistorted_img)

##RECTIFICATION
# Extrinsic μεταξύ cam0 και cam1 (από T_cn_cnm1)
R = Tc1c0[:3,:3]
t = Tc1c0[:3,3]
im_size=(512,512)
'''Κανει την ευθυγραμμιση στις επιπολικες γραμμες των 2 εικονων'''
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K0, None, K1, None, im_size, R, t, flags=cv2.CALIB_ZERO_DISPARITY)
##Οι πινακες P λενε που πανε τα πιξελ της παλιας εικονας στη νεα (mapping)
##Για pixel στη θέση (i,j) της rectified εικόνας, η τιμή του βρίσκεται στη θέση (map_x[i,j], map_y[i,j]) της αρχικής εικόνας.
##Κάθε pixel (u, v) της rectified εικόνας υπολογίζεται ως: rectified image (u,v)=original image(mapx[u,v],mapy[u,v])

# Νεοι πινακες K
K0_rectified = P1[:3, :3]
K1_rectified = P2[:3, :3]
output_calib_dir = "calibration_data_rectified/"
os.makedirs(output_calib_dir, exist_ok=True)

np.save(os.path.join(output_calib_dir, "K0_rectified.npy"), K0_rectified)
np.save(os.path.join(output_calib_dir, "K1_rectified.npy"), K1_rectified)
np.save(os.path.join(output_calib_dir, "P1.npy"), P1)
np.save(os.path.join(output_calib_dir, "P2.npy"), P2)
np.save(os.path.join(output_calib_dir, "Q_matrix.npy"), Q)


map0_x, map0_y = cv2.initUndistortRectifyMap(K0, None, R1, P1, im_size, cv2.CV_32FC1)
map1_x, map1_y = cv2.initUndistortRectifyMap(K1, None, R2, P2, im_size, cv2.CV_32FC1)

# Φάκελοι με undistorted εικόνες
left_images_dir = "cam0_images_undistorted/"
right_images_dir = "cam1_images_undistorted/"
output_dir = "rectified/"

#Για την απαριθμση και σωστη ταξινομηση των αποτελεσματων
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

os.makedirs(output_dir, exist_ok=True)
image_files = sorted(os.listdir(left_images_dir),key=natural_sort_key)

for idx, img_name in enumerate(image_files, start=1):
    left_img = cv2.imread(os.path.join(left_images_dir, img_name))
    right_img = cv2.imread(os.path.join(right_images_dir, img_name))
    
    left_rectified = cv2.remap(left_img, map0_x, map0_y, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, map1_x, map1_y, cv2.INTER_LINEAR)
    
    # Μορφοποίηση αριθμού με 5 ψηφία (π.χ., 00001)
    frame_id = f"{idx:05d}"
    cv2.imwrite(os.path.join("rectified", f"rect_left_{frame_id}.png"), left_rectified)
    cv2.imwrite(os.path.join("rectified", f"rect_right_{frame_id}.png"), right_rectified)

print(f"Ολοκληρώθηκε η επεξεργασία {len(image_files)} εικόνων!")
