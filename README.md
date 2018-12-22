# NO3sis


# Chapter 1 : blurring faces

It seems openCV is faster and as accurate as face_recognition. Both fail when the head is inclined

##  With openCV

First install openCV: 

```bash
pip install opencv-python
```

Then to perform live blurring on your webcam :

```bash
python blurring_opencv.py
```

Press 'q' to quit
## With face_recognition

[face_recognition](https://github.com/ageitgey/face_recognition) is a widely used package. 
To install it, you first need dlib :
```bash
git clone https://github.com/davisking/dlib.git
cd ..
python setup.py install
```

Now you can install face_recognition :

```bash
pip install face_recognition
```

And run

```bash
python blurring_dlib.py
```

Press 'h' for face recognition using HOG features, 'c' for CNN features (slower) and 'q' to quit



# Chapter 2 : face swapping

We will use the facial landmarks detected by dlib to do faceswap. To see what landmarks are run :

```bash
python face_landmarks.py
```

We first implemented 2 options
- faceswap with a fixed image 
```bash
 python face_swap_fixed_image.py image_path
 ``` 
 
 - faceswap with faces detected on the webcam (better with 2)
```bash
 python faceswap.py
 ``` 
 
 Faceswap is done by :
 - detecting faces 
 - finding associated landmarks
 - computing the homography between 2 faces using landmarks
 - warping a face to the other using homography
 - applying color normalization to the warped face