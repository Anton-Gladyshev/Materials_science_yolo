# Materials science Yolo
yolo v8 (you look only once) object detection network applied for atom tracking in Materials science

This library is essentially a wrapper that allows to track atoms in electron microscopy images. Here you can find weights that are sutable for fining small round blobs in an image. 

A full description of yolo is avalible at ultralytics website, please concider reading https://docs.ultralytics.com/models/yolov8/.

## Setting Up the Python Environment and Installing MSYOLO

To create a proper Python environment and install this package, you can use **conda**, **mamba**, or **micromamba**. With **conda**, use:

```bash
$ git clone git@github.com:Anton-Gladyshev/msyolo.git
$ cd msyolo
$ conda env create -f msyolo.yml
$ conda activate msyolo
$ pip install .
```


## Usage examples

Please look into the folder 'tutorials'.

## Creating your weights

If you find that the weight i got are not sutable for your data, try training yolo yourself.
To crate a train dataset i suggest following steps:
1. Go to https://www.makesense.ai and upload your images
2. Pick the atoms with the avalible tools (I created only a single class `atom`)
3. Export the data in zipped yolo format, you will get a .txt file with coordiantes for each image you uploaded
3.1 (Optional )If you don't want to use the https://www.makesense.ai, you can create your own .txt data. In this case, for each image you create a .txt file with 5 colomns:
label, x (fraction of total image), y (fraction of total image), width (fraction of total image), height (fraction of total image). The final file should look more-or-less like this:
```txt
0 0.211310 0.208672 0.072003 0.073171
0 0.396072 0.168022 0.072003 0.073171
0 0.492528 0.140921 0.080154 0.067751
...
0 0.541436 0.041328 0.066569 0.069106
```
4. Sort the images and .txt files
4.1 Create main folder `master_folder_name`. 
4.2 Within the `master_folder_name` you have to create folders `images` and `labels`.
4.3 Within both folders  `images` and `labels` you have to create folders `train` and `val`.
Summarizing steps 4.1-4.3:
```bash
$ mkdir master_folder_name
$ cd master_folder_name
$ mkdir images
$ mkdir labels
$ cd images
$ mkdir train
$ mkdir val
$ cd ../labels
$ mkdir train
$ mkdir val
```
5. Split your labels and images into two sets: the first one will be used for training, the second one for validation. Fill the folders with the data. You want to rename your images into 1.png 2.png etc according to labels you got.
6. Now you need to create a .yaml file
```bash
$ touch atoms.yaml
```
7. Open the `atoms.yaml` and paste following info. You only have to change the path to `master_folder_name`
```
path: path to the master_folder_name
train: images/train  
val: images/val  
test:  
# Classes
names:
  0: atom
```
9. Now you just have to start python a run following script:
```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')  ## here you can also take my weights as an initial guess
path_yaml='atoms.yaml' ## here you insert the path to the yaml file you created
results = model.train(data=path_yaml, epochs=2000, batch=3)#,device='mps')
```
10. Done! For more info, please reffer to https://yolov8.com/#what-is
