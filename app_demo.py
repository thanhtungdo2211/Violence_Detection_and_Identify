import streamlit as st
from PIL import Image
from detection import model2
from re_id import extract_feature, load_network, sort_img, data_transforms, ft_net, fuse_all_conv_bn, assign_ID, clear_cache
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import math
from collections import Counter
from torchvision import datasets
import os
import glob

import torch.nn as nn
import scipy.io

# Config detection 
model2_path = 'model/best_violence_det_02.pt'
threshold_iou = 0.1
threshold_conf = 0.5
imgsz = 640
threshold_nms = 0.4

# Config re_id
stride = 2
nclasses = 751
linear_num = 512
batchsize = 64
test_dir = "/media/tung/New Volume/Programing/ViolenceDetectionbyDrone/data/Market-1501/pytorch_PersonReID_infermydata/"
name = 'ft_ResNet50'
which_epoch = '300'
result_mat = "pytorch_result.mat"
output_dir = os.path.join(test_dir, './query/0005/')
result = scipy.io.loadmat(result_mat)
gallery_feature = torch.FloatTensor(result['gallery_f']).cuda()
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]
N_images = 10
threshold_id = 0.5


# Load model
model_detect = YOLO(model2_path)
model_structure = ft_net(nclasses, stride=stride, linear_num=linear_num)
model = load_network(model_structure, name, which_epoch)
model.classifier.classifier = nn.Sequential()
model = model.eval()
if torch.cuda.is_available(): 
    model = model.cuda()
model = fuse_all_conv_bn(model)

# Initialize Streamlit app
st.title("Violence Detection and Identify Person")
st.markdown('<b style="font-size: 35px;">Object recognition application</b>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption='Input Image', use_column_width=True)
    image_array = np.array(image)
    image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    # Save the image temporarily to read it with OpenCV
    temp_file = "temp_image.jpg"
    cv2.imwrite(temp_file, image_cv)

    ## Detection
    t0 = time.time()
    person_images, bbox_violence, id_person = model2(temp_file, model_detect, threshold_iou, threshold_conf,threshold_nms, imgsz, output_dir)
    if len(bbox_violence) == 0:
            st.markdown('<b style="font-size: 20px;">No acts of violence</b>', unsafe_allow_html=True)
    t1 = time.time()
    for bbox in bbox_violence:
            cv2.rectangle(image_array, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            st.markdown('<b style="font-size: 20px;">Violent behavior detected</b>', unsafe_allow_html=True)
            # Display the image
            st.image(image_array, caption="Violent behavior ", channels="RGB")

    ## RE-ID
    image_datasets = {
        'gallery': datasets.ImageFolder(os.path.join(test_dir, 'gallery'), data_transforms),
        'query': datasets.ImageFolder(os.path.join(test_dir, 'query'), data_transforms)
    }
    dataloaders = {
        'gallery': torch.utils.data.DataLoader(image_datasets['gallery'], batch_size=batchsize, shuffle=False, num_workers=16),
        'query': torch.utils.data.DataLoader(image_datasets['query'], batch_size=batchsize, shuffle=False, num_workers=16)
    }
    with torch.no_grad():
        query_feature = extract_feature(model,dataloaders['query'],linear_num, batchsize)
    query_feature = query_feature.cuda()
    for i in id_person:
        index = sort_img(query_feature[i], gallery_feature, gallery_label, gallery_cam)
        query_path, _ = image_datasets['query'].imgs[i]
        result_of_query = []
        st.write(f'Identification results for the individual {i}:')
        cols = st.columns(11)
        # Display the image of the person cropped in the first column
        person_image = Image.open(query_path)  # This example assumes you have cropped the image of the person and saved it to `query_path`
        cols[0].image(person_image, caption='')
        
        # Display the next 10 Re-ID result images
        for j in range(N_images):
            img_path, _ = image_datasets['gallery'].imgs[index[j]]
            label = gallery_label[index[j]]
            result_of_query.append(label)
            image = Image.open(img_path)
            cols[j+1].image(image)  # Add 1 because the first column is already used to display the person image

        id = assign_ID(result_of_query, N_images, threshold_id)
        if id == -1:
            id = 0
        st.write(f'ID of the individual: id = {id:04}')
    t2 = time.time()
    total_time_violence = t2 - t0
    total_time_non_violence = t1 - t0
    if len(bbox_violence) == 0:
        st.write(f"Total processing time: {total_time_non_violence:.2f} seconds")
    else:
        st.write(f"Total processing time: {total_time_violence:.2f} seconds")
    clear_cache(output_dir)
