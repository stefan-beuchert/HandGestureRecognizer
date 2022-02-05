# HandGestureRecognizer


## Introduction and goal

## Theory (Stefan) 
Stefan, ich habe dir das Thema hier mal gegeben, weil ich glaube, dass du am meisten in der Theorie drin steckst.
Wir brauchen den Teil nicht so aufzublähen. Wir könnten hier ein bisschen auf bereits bestehende Lösungen eingehen.
Außerdem haben wir hier etwas Platz, um beispielsweise die mediaPipe zu erklären. Dann brauchst du das unten nicht 
mehr machen. - Matthias

Recognizing hand gestures is a classic computer vision use case and can be used to tackle many different problems, all of which can be categorized under the topic "human-machine interaction" or HMI. Using visual input to interact with machines in a harsh environment (underwater or in loud areas) where voice or physical inputs are not possible or just using gestures to trigger a camera are potential use cases where gesture recognition can be applied.

When talking about gesture recognition, the process from input to actual recognition can be divided into two subproblems:
The identification of the hand itself
The classification of the gesture that the hand is doing. This also includes the class "not a gesture" if the input does not match any previously learned patterns.

The solution in this repository makes use of the mediapipe library made by Google, which is one of the most widely shared and re-usable libraries for media processing. This pre-trained solution already tackles the first subproblem (the identification of the hand(s) itself) with a pre-trained ml model. In addition to that, it is also possible to represent the hand not as a sum of pixel value, but as a collection of 21 landmarks that define the hand. The tip of the pinky finger represented by an X, Y and Z-coordinate is one example of those landmarks. The different values of the coordinates are estimated by another pre-trained model contained in the mediapipe library. 

With the different hand landmarks as a database, it was possible to train a model to predict different hand gestures. The output was then used to interact hands-free with the classic atari breakout game as a showcase where this kind of solution can be applied.

## Data Set (Stefan)
Stefan, ich habe dir auch gleich das Thema hier gegeben, weil du den Datensatz ja auch bearbeitet hast.
Da ist es wahrscheinlich einfach, wenn du ein paar KeyFacts dazu nennst. Ich habe dir im Folgenden auch schon 
ein paar Ideen für insights dazugeschrieben. Kannst du nutzen, musst du aber nicht. - Matthias

some short explanations:
- which data set did we use
- where did it come from
- how big is it / how is it presented (single frames of videos)
- which properties (how many people/how many gestures/ how often shown)
- show some of the frames

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

1. Convert data to gestures
As described above (see "Theory"), the library mediapipe was used to detect and convert the single frames of a gesture (as grayscale image) to the coordinates of the different landmarks of the hand. More specifically the sum of pixel values is converted to 21 points described by an X, Y and Z value. This not only reduces the amount of data drastically but also gets rid of most of the noise in the original image (for example all the background data). 

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

5. Saving the results as CSV
The resulting gestures are saved in the CSV format with one frame per row, described by the initial label of the gesture and 21 columns for each X coordinate of the 21 landmarks and corresponding 21 columns for the Y and Z coordinates. In total 127.000 rows are created.

Apply the gesture model to the data --> Get the CSV-File with ~90k rows and 21 cols
describe the pipeline:
- download
- media pipe
- what else did you do, Stefan?

Um die Übergänge sauber hinzubekommen, wäre es gut, wenn du als letztes die Struktur der CSV erklärst. - Matthias.
### 2) Splitting:
The data was randomly assigned into training and test data by a proportion of 80/20.

### 3) Model fitting:
As we tried to distinguish between several gestures, we fitted two classification models to the training: a neural network (NN)
and a support vector machine (SVM).
We decided for these two machine learning algorithms as we hoped that they would be able to deal not just with the
high dimensional data but also with non-linear structures inside if there were any.

F1 svm normal -> 0.723
F1 svm combined -> 0.858
accuracy svm normal -> 0.979
accuracy svm combined -> 0.983

F1 nn normal -> 0.820
F1 nn combined -> 0.904
accuracy nn normal -> 0.987
accuracy nn combined -> 0.979

####   1. Neural network (Vy)
####   2. Support vector machine (Matthias)
Next to the NN also a SVM should be fitted to have a comparison of different algorithms.
The SVM was automatically trained with the test data based on the radial basis function as kernel function. The SVM tackled
the data as a multiclass problem.

### 4) Implement a small application to show the working system (Matthias)
To test our model in the real world we decided to apply the better model on a small game. 
A break-out game seemed to be good for this. There are just two possible moves for the bar: go to the left or go to the right.
Hence there are also just two gestures needed to control the game.

## Implementation

Where have been the problems? What did we do differently according to the plan shown above?
### 1) preprocessing: (Stefan)
- something about the size of the data:
- difficulties of storing them
- usage of kubernetes cluster to preprocess

### 3) Model fitting

#### 1. Neural network (Vy)
#### 2. Support vector machine (Matthias)
The SVM was selected as a second model because it is able to also deal with high dimensionality. But we forgot that SVM
has problems by fitting lots of data. Not cause of the results but caused by the calculation time. It is not just taken linearly
into account but quadratic. This caused computational times of over an hour per run. This was not yet a real problem. But
it already slowed the development a bit. Fortunately, this algorithm doesn't have too many hyperparameters as NN for example.

### 4) Implement a small application to show the working system (Matthias)

## Results (Matthias)

What did we reach? How good are the algorithms?
####   a) by the confusion matrix for both algorithms / also each algorithm separately (Which gestures are recognized best? (VY / Matthias)
####   b) by a PCA in two dims with colored datapoints (Matthias)



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

