# 필요한 라이브러리 선언

from flask import Flask, render_template, redirect, url_for, request
import numpy as np
from PIL import Image as pillow
import matplotlib as plt
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import pydicom

from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

import cv2
import torch
import torch.nn as nn
import pickle


### 파이 토치 모델 선언
# UNET 1차 모델
class UNet1(nn.Module):
    def __init__(self):
        super(UNet1, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        # kernel_size=3, stride=1, padding=1, bias=True 부분은 predefine 되어 있다.

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x

# UNET 2차 모델

class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        # kernel_size=3, stride=1, padding=1, bias=True 부분은 predefine 되어 있다.

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x


### 손실 함수 정의

# 다이스 계수(핵심 요소)
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))



# 파이 토치 관련 함수

def ToTensor(np_data):
    np_data = np_data.transpose(0,3,1,2)
    ts_data = torch.tensor(np_data,dtype=torch.float32)
    return ts_data

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)


### 전처리 함수(DICOM을 IMAGE로 변환)

def transform_to_hu(medical_image, image):
    hu_image = image * medical_image.RescaleSlope + medical_image.RescaleIntercept
    hu_image[hu_image < -1024] = -1024
    return hu_image

def window_image(image, window_center, window_width):
    window_image = image.copy()
    image_min = window_center - (window_width / 2)
    image_max = window_center + (window_width / 2)
    window_image[window_image < image_min] = image_min
    window_image[window_image > image_max] = image_max
    return window_image

def resize_normalize(image):
    image = np.array(image, dtype=np.float64)
    image -= np.min(image)
    image /= np.max(image)
    return image

def read_dicom(path, window_widht, window_level):
    print(path)
    image_medical = pydicom.dcmread(path)
    image_data = image_medical.pixel_array

    image_hu = transform_to_hu(image_medical, image_data)
    image_window = window_image(image_hu.copy(), window_level, window_widht)
    image_window_norm = resize_normalize(image_window)
    return image_window_norm

### 플라스크 시작!

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './images'
## 모델 선언
model = pickle.load(open('knn_ML_model.pkl', 'rb'))

# 메인 페이지 연결
@app.route("/")
@app.route("/main")
def main():
    return render_template('메인화면.html')

# 고객 로그인 페이지 연결
@app.route("/login_customer")
def customer_login():
   return render_template('고객용 로그인.html')

# 원하는 진단을 선택해주는 페이지 연결
@app.route("/select_customer")
def select_customer():
   return render_template('고객 선택.html')

# 진단을 위한 고객 페이지 연결
@app.route("/for_diagnosis")
def for_diagnosis():
   return render_template('진단 전 고객화면.html')

# 기존 등록한 CT 사진 진료 결과 목록 페이지 연결
@app.route("/check_diagnosis")
def check_diagnosis():
   return render_template('의사 진단 후 고객화면.html')

# 진단 요청 승인 페이지 연결
@app.route("/accept_request")
def accept_request():
   return render_template('진단 요청 승인.html')

# 의사 CT 진료 결과 페이지 연결
@app.route("/result_request")
def result_request():
   return render_template('CT 진단 결과.html')

# 의료진용 로그인 페이지 연결
@app.route("/doctor_login")
def doctor_login():
   return render_template('의료진용 로그인.html')

# 의료관계자과 확인할 수 있는 진료받을 고객 환자 목록 페이지 연결
@app.route("/list_customer")
def list_customer():
   return render_template('고객 나열 의사 화면.html')

# 환자 CT 사진을 진료할 수 있는 의료진용 페이지 연결
@app.route("/CT_diagnosis")
def CT_diagnosis():
   return render_template('CT 진단 화면.html')

@app.route("/ML_predict")
def ML_predict():
   return render_template('머신러닝.html')

# 진단 등록 완료 페이지와 연결
@app.route("/complete_diagnosis")
def complete_diagnosis():
   return render_template('진단 등록 완료.html')

# 고객이 dicom을 업로드하면 jpeg로 변환하고 predict하는 함수
@app.route('/show_dicom_and_predict', methods=['POST'])
def show_dicom() :
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        dicom = read_dicom("C:/Users/hyunj/PycharmProjects/hippo_web/images/"+filename,310,30)
        img = pillow.fromarray(dicom*255)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save("C:/Users/hyunj/PycharmProjects/hippo_web/static/img/dicom.jpeg")

        # predict 하는 부분!!
        dicom = np.expand_dims(dicom, axis=2)  # (512, 512, 1)

        X_test = np.zeros((1, 512, 512, 1), dtype=np.float32)

        X_test[0] = dicom
        X_test = ToTensor(X_test)
        # # Unet
        # with torch.no_grad():
        #     # forward
        #     net3.eval()
        #
        #     unet_output = net3(X_test) > 0
        #     unet_output = fn_tonumpy(unet_output) * 1
        #
        #     unet_output = unet_output * fn_tonumpy(X_test)
        #
        # # unet_output = fn_tonumpy(unet_output)
        #
        # dicom = fn_tonumpy(X_test)
        # dicom = np.concatenate([dicom[0], dicom[0], dicom[0]], axis=2)
        # dicom = pillow.fromarray((dicom * 255).astype('uint8'), 'RGB')
        #
        # kidney_mask = unet_output[0, :, :, 0] > 0
        # kidney_mask = kidney_mask * 1
        # kidney_mask = np.expand_dims(kidney_mask, axis=2)
        # kidney_mask = np.concatenate([kidney_mask, kidney_mask * 255, kidney_mask], axis=2)
        # kidney_mask = pillow.fromarray((kidney_mask).astype('uint8'), 'RGB')  # color image
        #
        # tumor_mask = unet_output[0, :, :, 1] > 0
        # tumor_mask = tumor_mask * 1
        # tumor_mask = np.expand_dims(tumor_mask, axis=2)
        # tumor_mask = np.concatenate([tumor_mask * 255, tumor_mask, tumor_mask], axis=2)
        # tumor_mask = pillow.fromarray((tumor_mask).astype('uint8'), 'RGB')  # color image
        #
        # full_image = cv2.add(np.array(dicom), np.array(kidney_mask))
        # full_image = cv2.add(full_image, np.array(tumor_mask))
        # full_image = pillow.fromarray(full_image)
        #
        # full_image.save("C:/Users/hyunj/PycharmProjects/hippo_web/static/img/pred.jpeg")

        # BLUnet
        with torch.no_grad():
            # forward
            net1.eval()

            output = net1(X_test) > 0
            output = fn_tonumpy(output) * 1

            output = output * fn_tonumpy(X_test)

        output = ToTensor(output)
        with torch.no_grad():
            net2.eval()
            output = net2(output)
        output = fn_tonumpy(output)

        dicom = fn_tonumpy(X_test)
        dicom = np.concatenate([dicom[0], dicom[0], dicom[0]], axis=2)
        dicom = pillow.fromarray((dicom * 255).astype('uint8'), 'RGB')

        kidney_mask = output[0, :, :, 0] > 0
        kidney_mask = kidney_mask * 1
        kidney_mask = np.expand_dims(kidney_mask, axis=2)
        kidney_mask = np.concatenate([kidney_mask, kidney_mask * 255, kidney_mask], axis=2)
        kidney_mask = pillow.fromarray((kidney_mask).astype('uint8'), 'RGB')  # color image

        tumor_mask = output[0, :, :, 1] > 0
        tumor_mask = tumor_mask * 1
        tumor_mask = np.expand_dims(tumor_mask, axis=2)
        tumor_mask = np.concatenate([tumor_mask * 255, tumor_mask, tumor_mask], axis=2)
        tumor_mask = pillow.fromarray((tumor_mask).astype('uint8'), 'RGB')  # color image

        full_image = cv2.add(np.array(dicom), np.array(kidney_mask))
        full_image = cv2.add(full_image, np.array(tumor_mask))
        full_image = pillow.fromarray(full_image)

        full_image.save("C:/Users/hyunj/PycharmProjects/hippo_web/static/img/pred.jpeg")

        return render_template("진단 전 고객화면.html")

# 넣은 건강검진 수치들을 보여주는 함수
@app.route('/predict_ML', methods=['POST'])
def predict_ML() :
    if request.method == 'POST':
        data1 = request.form['a']
        data2 = request.form['b']
        data3 = request.form['c']
        data4 = request.form['d']
        data5 = request.form['e']
        data6 = request.form['f']
        data7 = request.form['g']
        arr = np.array([[data1, data2, data3, data4, data5, data6, data7]])
        pred = model.predict(arr)
        return render_template('진단 전 고객화면.html', data=pred)

    return render_template( )

if __name__ == "__main__":
    net1 = UNet1()
    net1.load_state_dict(torch.load('bk_aug_model1.pt', map_location=torch.device('cpu')))

    net2= UNet2()
    net2.load_state_dict(torch.load('bk_aug_model2.pt', map_location=torch.device('cpu')))

    net3 = UNet2()
    net3.load_state_dict(torch.load('unet_model.pt', map_location=torch.device('cpu')))


    app.run(debug=True)
