# **Traffic Sign Recognition** 


---


[//]: # (Image References)



[imageA]: ./some_stuff/firstNen.jpeg "No_entry_1"
[imageB]: ./some_stuff/NenN.jpeg "No_entry_2"
[imageC]: ./some_stuff/Childre.jpeg "Children_crossing"
[imageD]: ./some_stuff/Stop.jpeg "Stop"
[imageE]: ./some_stuff/30kmh.jpeg "Speed_limit_30kmh"

[image1]: ./some_stuff/Sep_Conv.jpeg "Separable Convolution"
[image2]: ./some_stuff/Typ_Conv.jpeg "Typical Convolution"
[image3]: ./some_stuff/Data_Bars.jpeg "Input Data Bar Graph"
[image4]: ./some_stuff/Flow.jpeg "Flow"


---


### 1. Data Set Summary & Exploration


The code for this step is contained in the second code cell of the IPython notebook.  

* The size of training set is 34799x3
* The size of test set is 12630x3
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43


The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set, a bar graph of training test showing that for some traffic signs there are more than double as many training sets as for others signs (1500 vs 7000):

![alt text][image3]

I decided not to change these propotions in order to see the implications in the model accuracy.

### 2. Design and Test a Model Architecture

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to normalize the images to values between 0-255 with cv2.normalize(  ). Then the images were converted to grayscale with cv2.cvtColor(  ) and then to binary black and white with cv2.adaptiveThreshold(  ). The idea was not to reduce the data from 32x32x3 to 32x32x1 but to expand it to 32x32x5. 

The following image shows processing for the a new web image found in google's streetview (original data set have same processing):

![alt text][image4]



The difference between the original training data set and the augmented data set is 34799x3 vs 34799x5. Validation and test data set have similar diference.

I decided to augment the data sets in this way so that Neural Network could make a better generalization, by relating five versions of the same image with separable convolutional neural networks.

The code for my final model is located in the fifth cell of the ipython notebook. 

The final model is a LeNet adaptation as shown in the following table:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x5 (RGB + Gray Scale + Black & White)    | 
| SepConv 5x6x30        | 1x1 stride, valid padding, outputs 28x28x30 	|
| RELU					|												|
| Average pooling    	| 2x2 stride,  outputs 14x14x30 				|
| SepConv 30x4x120      | 1x1 stride, valid padding, outputs 10x10x120  |
| RELU	             	|            									|
| Average pooling		| 2x2 stride,  outputs 5x5x120      			|
| Flatten               | outputs 3000                                  |
| Fully connected		| outputs 516					     			|
| RELU					|												|
| Fully connected       | outputs 258                                   |
| RELU                  |                                               |
| Fully connected       | outputs 43                                    |



The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used a batch size of 16, 15 epochs and 0.0003 as learning rate.

The code for calculating the accuracy of the model is located in the sixth cell of the Ipython notebook.

My final model results were:

* validation set accuracy of 0.961
* test set accuracy of 0.943

The final architecture was derived from LeNet following the following iterative approach:

- baseline was the LeNet original architecture with 0.91 accuracy at the tenth traning epoch and 32x32x3 images as input.
- I began iterating reducing learning rate and adding epochs with augmented input of 32x32x6. (5 iterations)
- then I reduced the input to 32x32x5, and iterated with learing rates and epochs (5 iterations)
- then I changed to 1:6@input and 6:1@output and iterated with learing rates and epochs (5 iterations)
- then I changed from typical convolutions + max pools to separeble convolutions + average pools and iterated with learning rates and epochs (5 iterations)
- then I reduced the batch size and iterated learning rates and epochs (5 iterations)

The most important design choices were the number of input channels (and thus the use of separable convulotions) and the input to output ratio.

I decided to use Separable Convolutions to enable the Network to manage five channels at the input (LeNet originally had only one channel) without convoluting depthwise with all the channels at the early stage in order to reduce parametric complexity. Before taking this action the 32x32x5 input model was having an accuracy of 88%, much less than baseline Lenet 32x32x3 input model (91%).

Poolings were changed from max to average as advised in the lessons, but only after the 35x35x5 got around 90% accuracy with separable convolutions.

Finally I tried to keep the propotions of the orginal LeNet at the first and last layers; from 1:6 in the input and 8.4:1 (84:10) in the output to 1:6 in the input and 258:43 (6:1) in the output. 

(1:6):(6:1) looks much nicer than (1:6):(8.4:1) though.

![alt text][image1]

[Snapshot from Vincent Vanhoucke's "Learning Visual Representations at Scale" pg.30](https://bcb9f395-a-8ac90a7d-s-sites.googlegroups.com/a/vanhoucke.com/vincent/publications/vanhoucke-iclr14.pdf?attachauth=ANoY7cqY_8tpwUHshUqxJvPUlQDMOlkbVWG_XbfTAXLAX-w0rxGk4eZ6yd7OPpkzHx2mmTLiWLEjtuQn3fQLymqR2MWPTToCk3w1SmGQCGxgRYx2xyuKJNbzFAqkXHygHXW8Uf5DGfQympO7TT7oD9qKfaU6n8vPisAtVdBPxiZatXNAEAI5JcZ3ag1X_LIdIm2qpdXJ4tjnX7jVzJvZh4ie1KnhzglFlYu2mKLj_XmK6drjMviS2Ks%3D&attredirects=0)




### 3. Test a Model on New Images


Here are five German traffic signs that I found on the web:


The first image is a ["No entry" sign at Stuttgart](https://www.google.com.mx/maps/@48.7745537,9.17997,3a,75y,2.78h,81.28t/data=!3m6!1e1!3m4!1sGgpEUooDeH_TX7byCR39AQ!2e0!7i13312!8i6656) that leaves an empty space in the bottom.

![alt text][imageA]

The secong image is an [occluded "No entry" sign at Stuttgart](https://www.google.com.mx/maps/@48.7702273,9.1439989,3a,89.8y,57.44h,82.42t/data=!3m6!1e1!3m4!1s6wa2i8dUsj25bspOSr21_w!2e0!7i13312!8i6656) that might be difficult to classify because almost a third of it is covered with leaves.

![alt text][imageB]

The third is a ["Children crossing" sign at Stuttgart](https://www.google.com.mx/maps/@48.769348,9.1630426,3a,75y,57.15h,80.65t/data=!3m6!1e1!3m4!1ssC0vJJwE2KrcSwklmjezMw!2e0!7i13312!8i6656)

![alt text][imageC]

The fourth is a ["Stop" sign at Stuttgart](https://www.google.com.mx/maps/@48.769432,9.1731114,3a,75y,115.84h,77.56t/data=!3m6!1e1!3m4!1siiQQZB9CXu9ZMNA4IOVoRQ!2e0!7i13312!8i6656) that is tilted and leaves an empy space in the bottom.

![alt text][imageD]

The las picture is a [30kmh Speed limit sign at Stuttgart](https://www.google.com.mx/maps/@48.772999,9.1838682,3a,34.5y,333.75h,90t/data=!3m6!1e1!3m4!1srFdGx9jifw9-R_M6K6JjSw!2e0!7i13312!8i6656) that should not bring any problems.

![alt_text][imageE]


The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No entry  									| 
| No entry     			| No entry										|
| Children Crossing		| Children Crossing								|
| Stop	      		    | Stop			        		 				|
| Speed limit (30km/h)	| Speed limit (30km/h)							|


The model was able to correctly guess all of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.3%


The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure that this is a "No entry sign" (probability of 1.00), and the image does contain a stop sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No entry  					    			| 
| 9e-16                 | Stop                                          | 
| 2e-16                 | Speed limit (20km/h)	     			     	|
| 3e-19		            | Bicycles Crossing  			 				|
| 2e-19         	    | Keep right             						|


For the second image, even with a 1/3 occlusion, the model predicted it with 93% certanty.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9333       			| No entry  					    			| 
| 0.0606                | No passing                       				|
| 0.0025   	            | Priority Road	  		                		|
| 0.0024	            | End of no passing    			 				|
| 0.0005       	        | Children crossing        						|


For the third image it was pretty sure it was a Children Crossing sign even when it was the sign with least training images (around 1,200 samples) vs other signs (over 7,000 sample sets)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Children Crossing  			    			| 
| 1e-4                  | Dangerous curve to the left     				|
| 1e-5    	            | Ahead only 	                 				|
| 1e-6		            | Slippery road             	 				|
| 1e-7         	        | Dangerous curve to the right    				|



For the fourth image the model was correct althou not pretty sure, both sign had around the same number of trainnning sets 3,500 sample sets. The model could differentiate shape (octagon vs. circle) and lines (single horizontal line  vs horizontal S T O P lettes)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.63280      			| Stop              			    			| 
| 0.36581               | No Entry                      				|
| 0.00070  	            | Yield                         				|
| 0.00060	            | Priority Road               	 				|
| 0.00003      	        | Speed limit (20km/h)                       	|



For the fifth image the model was not so sure it was right, I think this is due to the fact that more than one fourth of the 43 classes are of the same type (round red circle with black figures in the center).

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.550      			| Speed limit (30km/h)    		    			| 
| 0.300                 | Bicycles crossing                      		|
| 0.135   	            | Speed limit (20km/h)            				|
| 0.009  	            | Keep right                	 				|
| 0.005      	        | Road Work                                 	|







