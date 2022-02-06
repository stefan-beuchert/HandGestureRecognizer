# HandGestureRecognizer

### Table of Contents (Vy)

## Introduction and goal (Vy)

## Theory (Stefan) 
Recognizing hand gestures is a classic computer vision use case and can be used to tackle many different problems, all of which can be categorized under the topic "human-machine interaction" or HMI. Using visual input to interact with machines in a harsh environment (underwater or in loud areas) where voice or physical inputs are not possible or just using gestures to trigger a camera are potential use cases where gesture recognition can be applied.

When talking about gesture recognition, the process from input to actual recognition can be divided into two subproblems:
The identification of the hand itself
The classification of the gesture that the hand is doing. This also includes the class "not a gesture" if the input does not match any previously learned patterns.

The solution in this repository makes use of the mediapipe library made by Google, which is one of the most widely shared and re-usable libraries for media processing. This pre-trained solution already tackles the first subproblem (the identification of the hand(s) itself) with a pre-trained ml model. In addition to that, it is also possible to represent the hand not as a sum of pixel value, but as a collection of 21 landmarks that define the hand. The tip of the pinky finger represented by an X, Y and Z-coordinate is one example of those landmarks. The different values of the coordinates are estimated by another pre-trained model contained in the mediapipe library. 

With the different hand landmarks as a database, it was possible to train a model to predict different hand gestures. The output was then used to interact hands-free with the classic atari breakout game as a showcase where this kind of solution can be applied.

## Data Set (Stefan)
The data used to train the final model(s) was provided by Alessandro Floris and his team, who introduced it in a paper called "A dynamic hand gesture recognition data set for human-computer interfaces"

The data set consists of 27 classes where each class contains a different dynamic hand gesture. For each class, 21 different participants performed the gesture 3 three times in front of a white wall. All the participants have been trained in how to perform the specific gesture and wrongly executed footage has been excluded from the data set. Also, all the gestures were made from a perspective that would resemble a person sitting in front of a laptop camera.

This totals a collection of 1701 different videos (63 for each gesture and) stored as HD images in the png format for each frame. The roughly 245 GB of data is stored in multiple .zip files on the IEEE Data Port (on an underlying AWS S3 service).

## Concept and Design
We designed a data pipeline (Is this the right name???????) consisting of four elements:
1. Preprocessing: Application of Googles MediaPipe Hands to the images
2. Splitting: in training and test data
3. Model fitting: NN and SVM
4. Application of fitted models on a small game

These four steps are further explained in the following:

### 1) Preprocessing: (Stefan)
Before even starting the actual preprocessing, a solution had to be found to deal with the enormous data size. Handling 245 GB of data on a laptop was no option since the calculation would have taken several hours. Doing multiple iterations to improve the outcome with this limitation would have been a very painful process. Luckily we have the BHT cluster available where we could download the data to a PVC and interact with it using pods. This allowed a less time-intensive interaction with the data.

As described above (see "Theory"), the library mediapipe was then used to detect and convert the single frames of a gesture (as grayscale image) to the coordinates of the different landmarks of the hand. More specifically the sum of pixel values is converted to 21 points described by an X, Y and Z value. This not only reduces the amount of data drastically but also gets rid of most of the noise in the original image (for example all the background data). 

The resulting gestures are saved in the CSV format with one frame per row, described by the initial label of the gesture and 21 columns for each X coordinate of the 21 landmarks and corresponding 21 columns for the Y and Z coordinates. In total 127.000 rows are created.

Apply the gesture model to the data --> Get the CSV-File with ~90k rows and 21 cols
describe the pipeline:
- download
- media pipe
- what else did you do, Stefan?

Um die Übergänge sauber hinzubekommen, wäre es gut, wenn du als letztes die Struktur der CSV erklärst. - Matthias.
### 2) Splitting: (Vy)
The data was randomly assigned into training and test data by a proportion of 80/20.

### 3) Model fitting:
As we tried to distinguish between several gestures, we fitted two classification models to the training: a neural network (NN)
and a support vector machine (SVM).
We decided for these two machine learning algorithms as we hoped that they would be able to deal not just with the
high dimensional data but also with non-linear structures inside if there were any.

####   1. Neural network (Vy)
We fitted a Neural Network (NN) for the gesture recognition.

####   2. Support vector machine (Matthias)
Next to the NN also a SVM should be fitted to have a comparison of different algorithms.
The SVM is automatically trained with the training data based on the radial basis function as kernel function. It tackles
the data as a multiclass problem. This algorithm is selected as a second model because it is able to also deal with high 
dimensional data.

### 4) Implement a small application to show the working system (Matthias)
To test our model in the real world we decided to apply the better model on a small game. 
A break-out game seemed to be good for this. There are just two possible moves for the bar: go to the left or go to the right.
Hence there are also just two gestures needed to control the game. The whole pipeline should be kept easy and is orientated 
on the steps seen above. The camera should take a picture. As within the preprocessing part, the image is analysed by the
mediapipe and the coordinates are then passed to the forward-path of the NN or the SVM to take a prediction of the assumed
class. According to the prediction the bar is then moved to the right or left as long as the gesture is seen. If the image
is not classified as one of the gestures above the bar is also not moving.

## Implementation

Where have been the problems? What did we do differently according to the plan shown above?
### 1) preprocessing: (Stefan)
1. PreProcessing
Mediapipe was used to detect the hand in a frame and extract specific landmarks from the gesture. Those landmarks are represented by 3D coordinates and are the database for the model.

The values for the X and Y values are automatically scaled between 0 and 1 corresponding to the original size of the image. Therefore also the position of the landmark (and hand) is still included in the coordinates. For the Z coordinate, the mediapipe model calculates a value relative to the landmark of the bottom of the palm (which is therefore assigned the baseline value "0"). Possible values for Z range between very small numbers like 0.001 and bigger like 180 which makes them hard to work with, since the scale seems off. Further information on handling the initial scaling is in the following parts.

2. Exclude non-existing rows
Before continuing it is wise to get rid of the observations where mediapipe was not able to detect a hand or detected multiple hands. The according frames are not looked at further and are just skipped. 

3. Scaling to simulate bounding boxes
The automatic scaling of the mediapipe output for the X and Y coordinates is problematic for the use case, since it does not give the relative value of the landmarks to each other, but the value of the landmark in relation to the position in the image. In other words, the same gesture once done in the top right of a frame and once in the bottom left of another frame will result in drastically different coordinate values, even if both gestures are identical. The remove the noise of the positional difference, another scaling was applied.

This second scaling of the data does not look at the whole image, but only at the extracted landmarks. More specifically at the X, Y and Z coordinates over all the landmarks of one frame separately. They are then scaled from values of 0 (for the smallest value) and 1 (for the biggest). This is one way of extracting the hand from the full image and is a simpler version of the related approach of using bounding boxes.

One example for more clarity:
Two separate frames have the same hand gesture, where a person once makes a thumbs-up gesture in the top left and once in the bottom right of an image. Even when both gestures are identical, the coordinates will very much differ, since the gesture in the top left will have X and Y values closer to one and the gesture in the bottom right closer to 0. Be rescaling the X values between the lowest X coordinate of all the landmarks in one frame (probably the tip of the thumb, since it is a thumbs-up gesture) and the highest X value (probably the bottom of the pinky / the pinky metacarpophalangeal) both frames will have the same values for each X coordinate. The position in the original frame is not important anymore. 

The same is done for the Y and Z coordinate.
Criticism: It could be said that this method will distort the relations of the landmarks since it will be always scaled between 0 and 1 for each dimension separately. More specifically a hand gesture of a raised hand with fingertips sticking together and a gesture of a raised hand with fingers spread apart will result in the same coordinates after rescaling since the X values will always be between 0 and one for thump to pinky.

4. Combination of similar classes to one bigger class
After a manual inspection of the different gestures, it was clear that some of them do not differ at all when looking at the single frames. With some gestures, it was even hard for the human inspector to separate a gif into one of two classes when looking at them. 

To tackle this problem a selection of similar gestures (when only looking at the single frames) was created and given the same label. With this, the number of different classes was reduced from 27 to 17. It is important to mention that only classes with overall comparable frames (like waving from right to left, vs waving from left to right) have been combined. More complex gestures where only a part of the frames would fully match another class have not been added to a new collection. 

For further work on this task, it could be useful to (instead of a manual inspection) cluster the single frames by a clustering algorithm and then create new labels based on the insight of some samples of each cluster.
---> hier beispiel gifs einblenden

5. Storing data as CSV
The resulting gestures are saved in the CSV format with one frame per row, described by the initial label of the gesture and 21 columns for each X coordinate of the 21 landmarks and corresponding 21 columns for the Y and Z coordinates. In total 127.000 rows are created.

- something about the size of the data:
- difficulties of storing them
- usage of kubernetes cluster to preprocess

### 3) Model fitting (Vy)
We did stuff. We also combined some classes.
Vy, du kannst hier auch noch was darüber schreiben, dass wir die Modelle nach dem Training in Dateien schreiben. Das hat 
zumindest für mich die Entwicklungszeiten für die Auswertungs-und Visualisierungsalgorithmen beschleunigt. Außerdem wäre 
die Anwendung sonst nicht möglich gewesen. Ich werde unten dann hierauf verweisen. VG Matthias 

#### 1. Neural network (Vy)
#### 2. Support vector machine (Matthias)
But we forgot that SVM has problems by fitting lots of data. Not cause of the results but caused by the calculation time. 
It is not just taken linearly into account but quadratic. This caused computational times of over an hour per run. These times 
appeared to happen with a first demo data set containing fewer variables. So the computational times were reduced by taking
the real dataset which had more dimensions and was therefore easier to process for the SVM and also by reassigning the class
variables from strings to integers. This accelerated the whole training time from an hour down to just 5 Minutes.
But also the long training times before haven't been a real problem. The development has been slowed down. 
Fortunately, this algorithm doesn't have too many hyperparameters as NN for example. So there was not so much optimization needed.

### 4) Implement a small application to show the working system (Matthias)
As said before, we needed the saved models especially for the game. Nobody wants to play it if the training of the NN or
the SVM takes several minutes. For that reason we just import one model at the beginning of the game and use that for the
predictions. This is then acceptable for the loading times of the game.
While developing the game we discovered that combining the classes had also a good effect on the game itself. Instead of 
choosing several classes per direction, one for each direction was enough. This improved the game play as the speed
of the game increased.


## Results (Matthias)
Different metrics were applied to get an overview how good the algorithms perform. First we did a PCA, then we generated the
confusion matrices and calculated metrics from it and last we had a short look on the execution times.

####   a) PCA for first impression
For a first impression on the data we did a principal component analysis to have a look if there are already classes
which could be easily separated.

![PCA](https://github.com/stefan-beuchert/HandGestureRecognizer/blob/main/figures/PCA.png)

The first two components represent almost 50% of the variance. So with displaying it in two dimensions, more than one half
of the variance cannot be explained. So it is also not surprising that we cannot clearly separate classes from each other.
But one can see that some clusters of classes are formed. And just few classes are spread around the whole area.

The combination of these formed clusters by just explaining 50% of the variance gives some hope that a good separation of 
the data with more dimensions being involved is possible.

####   b) Confusion matrix to find similar gestures
To get a good impression on how well the algorithms perform for the different classes we generated the confusion matrices.
Values on the diagonal should be high as they mean "The algorithm predicted a given input vector as belonging to the class
and it actually belonged to the class". On the opposite, values in the rest of the matrix should be low as these mean "Given a vector
the prediction was wrong". The colors are according to it. 

![Conf_matrix_svm](https://github.com/stefan-beuchert/HandGestureRecognizer/blob/main/figures/svm_heatmap_27.png)
In the figure above, one can see clearly several clusters of classes where the SVM algorithm mispredicted some gestures.
For example, the classes one to three or the classes 20 and 21 form this kind of clusters. This plot was the basis of our
manual collections of class as mentioned above. But in general, it can be stated that there are also classes which can be 
classified quite precisely as classes 24 and 25.

![Conf_matrix_svm_combined](https://github.com/stefan-beuchert/HandGestureRecognizer/blob/main/figures/svm_heatmap_combined.png)
In the version where the SVM algorithm was run on the combined classes, one can see that some classes could be classified
better as class 16 but as we did it manually, one can also see that there are still collections which should be combined 
even further like the collections one and two. 

![Conf_matrix_nn](https://github.com/stefan-beuchert/HandGestureRecognizer/blob/main/figures/NN_heatmap_27.png)
For the NN confusion matrix a similar figure is shown. It also shows the same clusters of classes which should be combined to 
collections. But in comparison to the first figure, it can be easily stated that the overall classification is better as
there are lot less misclassified predictions compared to the SVM. So the NN performs better on this task and this accords
to our results above.

![Conf_matrix_nn_combined](https://github.com/stefan-beuchert/HandGestureRecognizer/blob/main/figures/NN_heatmap_combined.png)
The results for the NN with the collections is again similar to the SVM but also like the non-combined NN better than its SVM
counterpart. Overall this is also not surprising as we would have expected that both algorithms perform better but the order would
not change. 

In total, these results give us a good understanding of the data. They show gestures which might be similar if they are just 
interpreted frame-by-frame. That the algorithms come to similar results which classes should be combined to collections also
make our results more reliable. We would have been surprised if the algorithms suggest combining different classes. Like this
they support each other.


####    c) Score to compare the algorithms
After generating some graphs, we also calculated some score to compare the algorithms with one number. We first tried the 
accuracy but as this is a multiclass
problem and if we calculate a one-vs.-all accuracy metric, we get unreliable results. In that case the values for the true
negatives are far higher than all the others. So we always get good results.

Instead we decided to use the F1-Score. This metric can also be used for imbalanced data sets and multiclass classification
as we did. Now we got more different values for the algorithms. They are shown below.

| Algorithm    | Accuracy | F1-Score |
|--------------|----------|----------|
| SVM          | 0.979    | 0.723    |
| SVM combined | 0.983    | 0.858    |
| NN           | 0.987    | 0.820    |
| NN combined  | 0.979    | 0.904    |

It is not surprising to us that the NN is better than SVM as it is also the more complex and adaptive model. At the same
time we are not surprised that classes which we combined manually by looking on the confusion matrices gave better results
after combining.


####    d) Execution times to calculate the necessary ressources
Last we also had a look on the Execution Times for Training and testing of the algorithms:

| Algorithm    | Training | Testing  |
|--------------|----------|----------|
| SVM          | 6:54 min | 4:10 min |
| SVM combined | 6:14 min | 2:20 min |
| NN           | 2:36 min | 0.20598s |
| NN combined  | 2:38 min | 0.20813s |


## Discussion

Did our plan work? Where have been the difficulties? What didn't work out? Did we reach the 
overall goal to implement a game played with gestures?

### to Implementation 1) (Matthias / Stefan)
usage of just one cluster and not done several kubernetes pods at the time
--> still worked out as it had to be done just once and the cluster did the media pipe 
stuff in one night. Kubernetes configuration was more complicated than we thought.

## Conclusion
short summary of key facts

## References

