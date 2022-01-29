# HandGestureRecognizer


## Introduction and goal

## Theory (Stefan) 
Stefan, ich habe dir das Thema hier mal gegeben, weil ich glaube, dass du am meisten in der Theorie drin steckst.
Wir brauchen den Teil nicht so aufzublähen. Wir könnten hier ein bisschen auf bereits bestehende Lösungen eingehen.
Außerdem haben wir hier etwas Platz, um beispielsweise die mediaPipe zu erklären. Dann brauchst du das unten nicht 
mehr machen. - Matthias

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

## Concept and Design
We designed a data pipeline (Is this the right name???????) consisting of four elements:
1. Preprocessing: Application of Googles MediaPipe Hands to the images
2. Splitting: in training and test data
3. Model fitting: NN and SVM
4. Application of fitted models on a small game

These four steps are further explained in the following:

### 1) Preprocessing: (Stefan)
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

#### 	1. Neural network (Vy)
####	2. Support vector machine (Matthias)
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
it already slowed the development a bit. Fortunately this algorithm doesn't have too many hyperparameters as NN for example.

### 4) Implement a small application to show the working system (Matthias)

## Results (Matthias)

What did we reach? How good are the algorithms?
#### 	a) by the confusion matrix for both algorithms / also each algorithm separately (Which gestures are recognized best? (VY / Matthias)
####	b) by a PCA in two dims with colored datapoints (Matthias)



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

