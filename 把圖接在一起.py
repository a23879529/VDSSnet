import os
from PIL import Image
import torchvision
from torchvision import transforms
import torch


def con(source, i):

    a = "H:\\GridDehazeNet Attention-Based Multi-Scale Network for Image Dehazing\\video\\"
    b = "H:\\測試\\segment_EP1\\video\\"
    c = "H:\\測試\\senpai_改_EP1_stop\\video\\"
    d = "H:\\測試\\2021\\video\\"
    e = "H:\\測試\\kernel123_back3_EP2_stop\\video\\"

    # a = "H:\\GridDehazeNet Attention-Based Multi-Scale Network for Image Dehazing\\"
    # b = "H:\\測試\\segment_EP1\\"
    # c = "H:\\測試\\senpai_改_EP1_stop\\"
    # d = "H:\\測試\\2021\\"
    # e = "H:\\測試\\kernel123_back3_EP2_stop\\"

    # o = "H:\\NYU_v2_test\\"
    o = "H:\\FreeVideoToJPGConverter\\"
    # gt = "F:\\NYU-2\\"

    # out = "H:\\連接後圖\\"
    out = "H:\\temp\\"


    img1 = Image.open(a + source + "\\" + str(i) + '.png')
    img2 = Image.open(b + source + "\\" + str(i+1) + '.jpg')
    img3 = Image.open(c + source + "\\" + str(i+1) + '.jpg')
    img4 = Image.open(d + source + "\\" + str(i+1) + '.jpg')
    img5 = Image.open(e + source + "\\" + str(i+1) + '.jpg')

    imgo = Image.open(o + source + "\\" + str(i) + '.jpg')
    # imgo = Image.open(o + source + "\\" + str(i) + '_hazed.ppm')
    # imggt = Image.open(gt + source + "\\" + str(i) + '.ppm')

    my_transforms = transforms.Compose([
    transforms.Resize([240, 320]),
    transforms.ToTensor()
    ])

    img1 = my_transforms(img1)
    img2 = my_transforms(img2)
    img3 = my_transforms(img3)
    img4 = my_transforms(img4)
    img5 = my_transforms(img5)

    imgo = my_transforms(imgo)
    # imggt = my_transforms(imggt)

    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    img3 = img3.unsqueeze(0)
    img4 = img4.unsqueeze(0)
    img5 = img5.unsqueeze(0)

    imgo = imgo.unsqueeze(0)
    # imggt = imggt.unsqueeze(0)


    # torchvision.utils.save_image(torch.cat((imggt, imgo, img1, img2, img3, img4, img5), 0),
    #                                          out + source + str(i) + ".jpg")

    torchvision.utils.save_image(torch.cat((imgo, img1, img2, img5, img4, img3), 0),
                                             out + source + str(i) + ".jpg")

# a = "H:\\proposed\\library_0001a"
a = "H:\\測試\\kernel123_back3_EP2_stop\\video\\Driving"
count = 0
for file in os.listdir(a):
    count += 1
# print(count)
for i in range(count):
    con("Driving", i)
# con("cafe_0001b", 120)
print("Don!eDone!")