import copy

import cv2 as cv2
import os
from matplotlib import pyplot as plt
import base64
from binascii import a2b_base64
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import viewsets
from django.core import serializers
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import layers, models, optimizers

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Concatenate, Dot, Lambda, Input, Dropout, \
    ZeroPadding2D, Activation, concatenate, BatchNormalization, Conv1D, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from tensorflow.keras import backend as K

from .serializers import clientSerializer
from .models import client
import pickle
# from sklearn.externals import joblib
import json
import numpy as np
from sklearn import preprocessing
import pandas as pd

siamese_net = keras.models.load_model(r"C:\Users\DELL\PycharmProjects\inkuisitor\clerk\model_34")

@api_view(['GET', ])
def clientdetails_view(request):
    if request.method == 'GET':
        clients = client.objects.all()
        serializer = clientSerializer(clients, many=True)
        return Response({"data": serializer.data}, status=status.HTTP_200_OK)


@api_view(['POST', ])
def verify_view(request):
    average=0
    clientname=""
    if request.method == 'POST':
        option = request.data['option']

        if option == "Base64":
            BverifiedImg = request.data['BverifiedImg'].replace("data:image/png;base64,", "")
            clientname = request.data['clientName']
            path_client_name_image = r"C:/Users/DELL/Desktop/" + clientname + ".png"
            with open(path_client_name_image, "wb") as f:
                f.write(a2b_base64(BverifiedImg))

        serializer = clientSerializer(data=request.data)
        data = {}

        if serializer.is_valid():
            clientName = client.objects.filter(clientName=serializer.validated_data['clientName'])
            img1 = ""
            img2 = ""
            img3 = ""
            verifiedimg = ""
            if option == "Image":
                verifiedimg = serializer.validated_data['verifiedImg']
                for i, item in enumerate(clientName):
                    if item.Bimg1 == None:
                        img1 = item.img1
                        img2 = item.img2
                        img3 = item.img3
                    else:
                        img1 = item.Bimg1
                        img2 = item.Bimg2
                        img3 = item.Bimg3

            else:
                verifiedimg = clientname + ".png"
                for i, item in enumerate(clientName):
                    if item.Bimg1 == None:
                        img1 = item.img1
                        img2 = item.img2
                        img3 = item.img3
                    else:
                        img1 = item.Bimg1
                        img2 = item.Bimg2
                        img3 = item.Bimg3


            images = []
            images += [str(img1), str(img2), str(img3)]
            images = [r"C:/Users/DELL/PycharmProjects/inkuisitor/media/" + i for i in images]
            images += [r"C:/Users/DELL/Desktop/" + str(verifiedimg)]

            img_h, img_w, img_ch = 150, 300, 1
            image_shape = (img_h, img_w, img_ch)

            kernel = np.ones((9, 9), np.uint8)  # default

            def preprocessor_img(path, image_shape):
                image = cv2.imread(path, 0)
                blured = cv2.GaussianBlur(image, (9, 9), 0)
                threshold, binary = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=30)
                contours, hierarchies = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                the_biggest_contour_by_area = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(the_biggest_contour_by_area)
                cropped = image[y:y + h, x:x + w]
                resized = cv2.resize(cropped, image_shape, interpolation=cv2.INTER_LANCZOS4)
                # resized_blured = cv2.GaussianBlur(resized, (9,9), 0)
                threshold, resized_binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return resized_binary

            def make_pairs(images):

                all_pairs = []

                test_image = images[-1]
                originals_images = images[:]

                for i, img in enumerate(originals_images):
                    x1 = img
                    x2 = test_image
                    all_pairs += [[x1, x2]]

                # all_pairs = list(itertools.combinations(images[:-1], 2))

                pairs = []

                for ix, pair in enumerate(all_pairs):
                    img1 = preprocessor_img(pair[0], (img_w, img_h))
                    img2 = preprocessor_img(pair[1], (img_w, img_h))
                    # img1 = cv2.imread(pair[0],0)
                    # img2 = cv2.imread(pair[1],0)
                    # img1 = cv2.resize(img1, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)
                    # img2 = cv2.resize(img2, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)
                    img1 = img1.astype('float32')
                    img2 = img2.astype('float32')
                    img1 /= 255
                    img2 /= 255
                    img1 = np.atleast_3d(img1)
                    img2 = np.atleast_3d(img2)
                    pairs.append([img1, img2])

                pairs = np.array(pairs)
                return pairs

            paris_images = make_pairs(images)
            prediction_prob = siamese_net.predict([paris_images[:, 0], paris_images[:, 1]])
            average = sum(prediction_prob) / len(prediction_prob)
            print(prediction_prob)
            print(average)


        else:
            data = serializer.errors
        return Response(average)


@api_view(['POST', ])
def createprofile_view(request):
    if request.method == 'POST':
        data = {}
        option=request.data['option']
        if option == "Image":
            serializer = clientSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save()
            else:
                data = serializer.errors

        else:

            Bimg1 = request.data['Bimg1'].replace("data:image/png;base64,", "")
            clientname = request.data['clientName']
            path_client_name_image_1 = r"C:/Users/DELL/PycharmProjects/inkuisitor/media/imgs/" + clientname + "_1.png"
            with open(path_client_name_image_1, "wb") as f:
                f.write(a2b_base64(Bimg1))

            Bimg2 = request.data['Bimg2'].replace("data:image/png;base64,", "")
            clientname = request.data['clientName']
            path_client_name_image_2 = r"C:/Users/DELL/PycharmProjects/inkuisitor/media/imgs/" + clientname + "_2.png"
            with open(path_client_name_image_2, "wb") as f:
                f.write(a2b_base64(Bimg2))

            Bimg3 = request.data['Bimg3'].replace("data:image/png;base64,", "")
            clientname = request.data['clientName']
            path_client_name_image_3 = r"C:/Users/DELL/PycharmProjects/inkuisitor/media/imgs/" + clientname + "_3.png"
            with open(path_client_name_image_3, "wb") as f:
                f.write(a2b_base64(Bimg3))

            serializer = clientSerializer(data=request.data)
            if serializer.is_valid():
                serializer.validated_data['Bimg1'] = r"imgs/" + clientname + "_1.png"
                serializer.validated_data['Bimg2'] = r"imgs/" + clientname + "_2.png"
                serializer.validated_data['Bimg3'] = r"imgs/" + clientname + "_3.png"
                serializer.save()
            else:
                data = serializer.errors

        return Response(data)
