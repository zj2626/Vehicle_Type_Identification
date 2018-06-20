BIT-Vehicle Dataset

Dataset Description:
The BIT-Vehicle dataset contains 9,850 vehicle images. There are images with sizes of 1600*1200 and 1920*1080 captured from two cameras at different time and places in the dataset. The images contain changes in the illumination condition, the scale, the surface color of vehicles, and the viewpoint. The top or bottom parts of some vehicles are not included in the images due to the capturing
delay and the size of the vehicle. There may be one or two vehicles in one image, so the location of each vehicle is pre-annotated. The dataset can also be used for evaluating the performance of vehicle detection. All vehicles in the dataset are divided into six categories: Bus, Microbus, Minivan, Sedan, SUV, and Truck. The number of vehicles per vehicle type are 558, 883, 476, 5,922, 1,392, and 822, respectively.

Annotation Information Description:
The file named "VehicleInfo.mat" which can be opend by matlab contains the annotation information. There is a struct array named "VehicleInfo" in the "VehicleInfo.mat". The "VehicleInfo" are size of 10400*1, and each element describes an image. The fields of the struct are as follows.
name: 		The filename of the image.
height:		The height of the image.
wideth: 	The width of the image.
nvehicles:	The number of the vehicles in the image.
vehicles: 	This field is a struct array with the size of 1*nvehicles, and each element describes a vehicle. Each element 				contains five fileds: left, top, right, bottom, and category. The former four fileds characterize the location of the 		vehicle in the image, and the field "category" represents the type of the vehicle.
 

mat_to_train.py: 把原图像集合分为简单的两个集合 一个train一个val 以及train.txt和val.txt 其中val集合有1500张图像
mat_to_train_with_position.py: 把原图像集合分为两个集合 一个train一个val 以及train.txt和val.txt 两个txt中包含图像详细信息