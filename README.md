# Take N Make
### Recipe Suggestions

This program takes an input of an image of the interior of a refridgerator and suggests 
a recipe for you based on the contents of your fridge. 

<img src="https://raw.githubusercontent.com/forlucas27/Images/main/Labeled%20Fridge%20Picture.jpg" width="500"/>

There are several steps involved in this:
  1. The image of the fridge is segmented into images of individual food items
  2. A model is trained on images of food items scraped from Google
  3. The images of individual food items are fed into the model for classification
  4. The labels are run through a couple APIs in order to get a url for a recipe suggestion

## How to Run the App Locally

1. Clone or download the repository
2. Navigate in terminal to Recipe_app directory
3. Install the requirements.txt using the command:
   ```pip install -r requirements.txt```
5. Run the app using the command:
   ```flask run```
7. Now you can input your fridge image and get recipe suggestions!

## Input Image

<img src="https://raw.githubusercontent.com/forlucas27/Images/main/Inside_Fridge_10.jpg" width="500" />

## Segmentation

In order to process the fridge image into its individaul components a couple processing 
techniques are used: 
  - K-means clustering 
  - Watershed Algorithm
  
Based upon the grouped objects, the image is broken into individual items:
 
<img src="https://raw.githubusercontent.com/forlucas27/Images/main/Segmented_Fridge_Images.JPG" width="500" />

## Model

The model thus far has been trained on ~56000 images (~17.5 GB) placed in one of 47 categories.

### Random Subset of the Training Images
<img src="https://raw.githubusercontent.com/forlucas27/Images/main/Training_Images.JPG" width="600" />

### Model Accuracy

The trained model is ~ 78% accurate overall. A confusion matrix for a subset of the images is
shown below where the food categories are represented by numerical values along the axes. 
Each row of the matrix represents the instances in a predicted class, while each column represents 
the instances in a true or actual class. 

<img src="https://raw.githubusercontent.com/forlucas27/Images/main/confusion_mat_47Cat.jpg" width="800" />

A legend detailing the numerical categories can be viewed [here](https://raw.githubusercontent.com/forlucas27/Images/main/Food_Categories.png). 

## Utilizing Model for Classification of Fridge Images

Once the model is trained the real data from the input segmented fridge image can be classified. 

<img src="https://raw.githubusercontent.com/forlucas27/Images/main/Labeled_Images%20(1).jpg" width="500" />

## Final Step- Get Recipe Suggestion

The output classification labels are then ran through two API's: (1) the spoonacular API to get
a recipe title from an input of a list of ingredients and (2) google custom search to get the 
actual recipe URL from the recipe title name.

[Output Recipe](https://themccallumsshamrockpatch.com/2016/05/08/grilled-chicken-salad-with-fruit/)

## Thank you

Contact me at: rachel.lucas@mg.thedataincubator.com
