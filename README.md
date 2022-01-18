# *VideoAnalysisML* - Video Analysis using Transfer Learning, ANNs and XGBoost

## Methodology
### Problem
Predict the screen time of characters in a video file.

### Development Flow
The general workflow to tackle this problem is to first know what kind of machine learning tasks need to be used. Since we are working with labeled data and trying to know if a character appears in an image, this problem is a supervised image classification task. Next, we need to build our dataset which will comprise extracted images from a video file and their respective labels in a csv file. The data is preprocessed before training. This step includes the use of Transfer Learning which is a technique to extract useful features of an image and facilitate image classification when we count with a small number of samples. After that comes training where we use Artificial Neural Networks and XGBoost for classification. We will evaluate the models using evaluation metrics and experiment until we are satisfied with their results. The final and more exciting part of this project is using the model. That is why, we will use the models that were trained on a small sample of images from the show to infer the screen time of characters in a totally new episode. 

<img src='https://github.com/kvelazco/VideoAnalysisML/blob/main/Extras/development%20flow.png' />

## Dataset and preprocess
### Image
- Clips from episode 1, 2, 5, 6, 7 [mp4, 1280x720p] 
- Image extraction using OpenCV, 1 image frame per 1 second of video
- 541 images extracted

<img src='https://github.com/kvelazco/VideoAnalysisML/blob/main/Extras/image.png' />


### Labels
- Manually labeled images
- Class 0 -> none (backgrounds, secondary characters)
- Class 1 -> Tanjiro
- Class 2 -> Nezuko

<img src='https://github.com/kvelazco/VideoAnalysisML/blob/main/Extras/label.png' />

### Preprocessing
<img src='https://github.com/kvelazco/VideoAnalysisML/blob/main/Extras/preprocess.png' />

#### Image Preprocessing
- Images to numpy array
- Resizing to 224x224
- Train test split
- Normalization

#### Transfer Learning
- Oxford’s VGG16
- Google’s InceptionV3

<img src='https://github.com/kvelazco/VideoAnalysisML/blob/main/Extras/feature%20maps.png' />

#### SMOTE Oversampling
- Balance data 

## Training and Evaluation
### Results:
<img src='https://github.com/kvelazco/VideoAnalysisML/blob/main/Extras/training%20and%20evaluation.png' />


## Inference 
- Repeat the same steps to process images
- Load video to test 
- Extract images
- Images to numpy array
- Resize
- Normalize
- Extract features
- Predict using trained models	

### Real Values
- Class 0: 825 seconds
- Class 1: 518 seconds
- Class 2: 139 seconds

### One of the Results
<img src='https://github.com/kvelazco/VideoAnalysisML/blob/main/Extras/inference.png' />

## Discussion
The results indicate that it is possible to predict the screen time of characters in video by using machine learning techniques. The combination of InceptionV3 and XGBoost also showed that XGBoost can beat an artificial neural network (VGG16 + ANN) in at least one of the classes tested (class 2). XGBoost took around ¼ of the time the neural network took to train. This shows that with the proper tuning, the XGBoost algorithm can be at par or surpass an ANN and use less processing power and time. 

One of the limitations of this project was making the process a single label image classification task from the start. Some images sometimes contained both character 1 and 2 in the same frame. I had to pick the most representative to me, making my dataset biased from the start. To improve accuracy and overall results, further research in this area should be done by making this a multilabel image classification task. 

## Libraries Used
- numpy
- pandas
- matplotlib
- math
- cv2
- seaborn
- keras
- skimage
- sklearn
- imblearn (SMOTE)
- xgboost
