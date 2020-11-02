import numpy as np
import cv2
from matplotlib import pyplot as plt
import string;
from io import StringIO;


## function use to read image
def readImage(url):
    width = 20
    height = 20
    image = cv2.imread(url,0);
    resized = cv2.resize(image, (width,height), interpolation = cv2.INTER_AREA);
    return resized;
# =================================================

# load dataset [0-9] & [A-Z] and store it to images array
def loadDataSet(sampleLength, imageLength):
    images = []
    for i in range (1,sampleLength + 1):
        arrTemp = []
        sampleName = '' + str(i);
        if(i < 10):
            sampleName = "0" + sampleName;
        for j in range (1, imageLength + 1):
            imageName = '' + str(j);
            if(j < 10):
                imageName = "0" + imageName;
            arrTemp.append(readImage(uri + pathSrc.format(sampleName, imageName)));
        images.append(arrTemp)
    return images;

# =================================================

# =================================================
def recognizeText(images, image_test_url, kNearSize):
    k = np.arange(sampleLength)
    np_images = np.array(images)
    train = np_images[:,:imageLength].reshape(-1, 400).astype(np.float32)
    train_lables = np.repeat(k, imageLength)[:, np.newaxis];

    image_test = readImage(image_test_url)
    test = image_test.reshape(-1, 400).astype(np.float32);
    knn = cv2.ml.KNearest_create()
    knn.train(train,cv2.ml.ROW_SAMPLE,train_lables)
    ret, results, neighbours, dist = knn.findNearest(test, kNearSize)
    return results.astype(np.int32).reshape(-1);

def main():
    # declaration constant
    global uri;
    global pathSrc;
    global sampleLength;
    global imageLength;
    trans = list(range(0,10)) + list(string.ascii_uppercase) + list(string.ascii_lowercase)
    uri = "dataset\\";
    pathSrc = 'Sample0{0}\img0{0}-0{1}.png';
    sampleLength = 62;
    imageLength = 55;
    images = loadDataSet(sampleLength,imageLength);
    # replace path image test
    image_test_url = uri + pathSrc.format('03', '01');
    result = recognizeText(images, image_test_url , 5);
    print(trans[result[0]])
main();












