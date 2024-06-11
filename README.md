# Image_classification and outfit recommedation

Introduction
To build a deep learning model that recognises a face and assigns a style. The style category is then mapped with the fashion dataset by using a machine learning algorithm to identify best fashion for the facial features.
Dataset
The dataset used in this project are celeba and fashion mnist. The celeba dataset is used to classify the style of the facial features and fashion mnist is to find the outfit for the style category.
Model Architecture
To classify the image for the style category resnet50 is used as the CNN architecture. To identify the style for each category I used KNN algorithm to classify the fashion images into groups.
Approach
The approach I took to complete this task was basic classification model build using transfer learning approach, this helps to get more accuracy than building a model from scratch. During training the model I trained the model to classify the output in three categories as Formal, Casual and Intellectual Appearance, this helps me to classify the images in these categories. Using these categories I was able to identify which style of clothing facial structure is categories.
Data Pre-processing
(category_data_preprocessing.py)
For data preprocessing in the celeba dataset I used certain features, which I think will
help me categories the images. Based on selected features I used some logical thinking and assigned a few categories based on the feature values. I also considered that the dataset should not become imbalanced so I categories the data to make it balanced and categories in the desired format. Then perform one-hot encoding to convert the categorical style features into numerical features.
In the fashion mnist dataset I could not understand the features of the dataset so I took every feature into consideration while building a machine learning model.
Model_building
(category_model.py) (recommedation_model.py)
The main objective is to categorise the images based on the facial features for which I chose resnet50 to build the model. For the loss function I used BCEWithLogitsLoss as it expects the output to be logistic, which helps in multiclass classification. And for optimizer I used adam optimiser, which is good for most use cases. I used a few hyperparameters to find the model which can give better results.
Then for classifying the fashion I used KNN where the images are converted into embeddings and then grouped based on its similarities and that group is labelled for the style category.
Model_evaluation
(category_model_evaluation.py)
When training the model with different parameters there will be few model outputs
and check points, we need to evaluate the model based on the accuracy metrics and train and test loss. This helps me to choose the best model for using in production.
*There is a folder model_evalavuation which contains the evaluation of the model.
For the KNN model we used the elbow method to identify the number of clusters, but in our use case we already know we need three clusters.
Data_pipeline
(final_out.py)
All the model requirements are loaded, when the input folder of the images is given
as the input it will generate an output.txt, which gives the output in the required format.
Conclusion
The model built to identify the style category is successful, to make it more accurate, more insights and other key metrics need to be taken place based on user needs. This is a rough example of what approach can be done to solve this problem statement.
