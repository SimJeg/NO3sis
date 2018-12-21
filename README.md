# NO3sis


# Warmup : blurring faces

It seems openCV is faster and as accurate as face_recognition. Both fail when the head is inclined

##  With openCV

First install openCV: 

```bash
pip install opencv-python
```

Then run :

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
python blurring_face_recognition.py
```

Press 'h' for face recognition using HOG features, 'c' for CNN features (slower) and 'q' to quit

