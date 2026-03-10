import json                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           #by gby
import os
import re
import cv2
import numpy as np
from utils.plots import plot_one_box
# import matplotlib.pyplot as plt
# from IPython.display import display, clear_output
from detector import Get_Pred
from PIL import Image
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np


def calRatio(x):
    x1,y1,x2,y2=x
    #print(x2-x1/y2-y1)
    return (x2-x1)/(y2-y1)


def resize_with_white_background(og_list,target_size=(224, 224), background_color=(255, 255, 255)):
    crop_list=[]
    for image in og_list:

    # Resize the image while maintaining aspect ratio
        
        image = Image.fromarray(image, 'RGB')
        image.thumbnail(target_size, Image.LANCZOS)
        

    # Create a new image with white background
        new_image = Image.new("RGB", target_size, background_color)

    # Calculate position to paste the resized image
        left = (target_size[0] - image.size[0]) // 2
        top = (target_size[1] - image.size[1]) // 2

    # Paste the resized image onto the white background
        new_image.paste(image, (left, top))

    # Save the new image
        crop_list.append(new_image)
    return crop_list

def crop(image, boxes_list):
    # 初始化一个空列表来存储裁剪后的图像
    cropped_images = []
    # 遍历所有的边界框
    i=0
    for box in boxes_list:
        # 解包每个边界框的坐标
        if(calRatio(box)>0.8):
            continue
        x1, y1, x2, y2 = map((int),box)

        # 裁剪图像
        cropped_image = image[y1:y2, x1:x2]

        # 将裁剪后的图像添加到列表中
        cropped_images.append(cropped_image)
        
        # square_image = np.array(pad_to_square(cropped_image))
        # cv2.imwrite(str(i)+"SquareWhite.jpg",square_image)
        # cv2.imwrite(str(i)+".jpg",cropped_image)
        # print(1)
        # i+=1

    return cropped_images




# def display_cropped_images(cropped_images):
   
#     # 定义每行显示的图像数量
#     IMAGES_PER_ROW = 4

#     num_rows = (len(cropped_images) + IMAGES_PER_ROW - 1) // IMAGES_PER_ROW
#     fig, axes = plt.subplots(num_rows, IMAGES_PER_ROW, figsize=(IMAGES_PER_ROW * 5, num_rows * 5))
#     for i, img in enumerate(cropped_images):
#         # 计算当前图像的行和列
#         row = i // IMAGES_PER_ROW
#         col = i % IMAGES_PER_ROW

#         axes[row, col].imshow(img)
#         axes[row, col].axis('off')  

#     plt.subplots_adjust(wspace=0.1, hspace=0.1)

#     # 显示整个图像网格
#     plt.show()

#     # 创建翻页功能
#     def next_page():
#         clear_output(wait=True)
#         display(fig)
#     def prev_page():
#         clear_output(wait=True)
#         display(fig)
#     plt.gcf().canvas.mpl_connect('key_press_event', lambda event: on_key(event, next_page, prev_page))

# def on_key(event, next_func, prev_func):
#     if event.key == 'right':
#         next_func()  # 右箭头键翻到下一页
#     elif event.key == 'left':
#         prev_func()  # 左箭头键翻到上一页


# Load the model
model = load_model(r"E:\vscodeP\AI\basketballAutoZooming\base_on_2021_Yolov5\keras_model.h5", compile=False)

# Load the labels
class_names = open(r"E:\vscodeP\AI\basketballAutoZooming\base_on_2021_Yolov5\labels.txt", "r").readlines()
size = (224, 224)
def print_confidence_score(final_list,cntFram):
    className=[]
    confidenceScore=[]

    i = 0
    for image in final_list:
        # Disable scientific notation for clarity
            np.set_printoptions(suppress=True)


    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
            

    # resizing the image to be at least 224x224 and then cropping from the center
            
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
            image_array = np.asarray(image)

    # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
            data[0] = normalized_image_array

    # Predicts the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            
            print(index)
            print(confidence_score)
            className.append(index)
            confidenceScore.append(confidence_score)
            i+=1
    return className,confidenceScore
    # Print prediction and confidence score
            # print("Class:", str(class_name[2:9]), end=" ")
            # print("Confidence Score:", str(confidence_score))
            # # 指定保存图像的文件夹路径
            
            # file_name = re.sub(r'[^a-zA-Z0-9]', '', (str(class_name[2:9])))+"__"+ re.sub(r'[^a-zA-Z0-9]', '', str(confidence_score))  +".jpg"    
            # print(file_name)      
            # cv2.imwrite(file_name,np.array(image))
                    
            

