# Satellite-based-Poverty-Estimation
Estimating Expenditure-based Poverty using Satellite Images

## Introduction
Poverty is a global-scale problem that concerns all countries, including Indonesia. Poverty, in a socio-economic context, is defined as the inability of a person of a household to meet their primary needs, i.e., clothing, shelter, and food. United Nations is committed to eradicating poverty by 2030, as stated in the first goals of Sustainable Development Goals (SDGs). Relevant data is needed to implement policies related to poverty eradication. There are five approaches to collecting poverty data: the financial approach, basic needs approach, capabilities approach, social exclusion approach, and rights approach. Most countries use household income or consumption as the basis for determining the welfare of the poor. Unfortunately, timely and reliable data collection is difficult and expensive because it requires detailed data collection directly from individuals and households.

New sources of potential data have emerged to estimate poverty, such as mobile phone data or satellite imagery. Recent developments in machine learning-based methods have enabled an intensive approach to measuring various poverty indicators. Satellite images comes in two forms, daytime satellite images and nighttime images. Daytime satellite images have a higher image resolution than nighttime satellite images. They provide plenty of helpful information in the form of colors and patterns. The approach taken in this work is replicated and modified from Jean's study (2016). This approach uses the Convolutional Neural Network (CNN) to determine the relationship between day and night satellite imagery. The CNN architecture used was retrained (fine-tuned) to predict the intensity of night light according to the input of daytime satellite imagery.

Using the Convolutional Neural Network (CNN) and different regression models, we wish to estimate the per capita expenditure of 40 regencies in Central Java and the Special Region of Yogyakarta. While fine-tuning VGG16, we also reduce the last fully connected layer to 32 nodes to avoid the curse of dimensionality. Since the data extracted from satellite images are not interpretable, we use Support Vector Regression (SVR), Extreme Gradient Boosting (XGBoost), and Random Forest. Out of the five poverty measurement approaches, the consumption expenditure-based monetary approach is the most appropriate response variable because only the residence dimension can be captured by satellite imagery. By using only public, accessible, and quickly updated data, the results of this study are expected to provide a grim picture but provide up-to-date information in mapping the level of per capita expenditure.

## Result
### Classifying Images
![image](https://user-images.githubusercontent.com/48485276/197674253-53cbb4cb-f96f-4e2b-b94c-472a2899bd0c.png)
We tried different method to lower the validation loss as shown in the picture above. Even though applying dropout produced a model with the lowest validation loss, the model didn't give convergent result. A model that did produce a convergent result was model with augmentation. It had almost the same minimum validation loss as the dropout model. But with more researh and time, it's possible to lower the validation loss even further.

![image](https://user-images.githubusercontent.com/48485276/197675251-62ee9031-4691-4da0-b304-d386ab854864.png)
As for the accuracy, dropout model, augment model, and the base model gave us the almost a convergence result. Looking at the graph, augment model is still the best one.

### Regression
#### Modeling
Using the relu activation function caused some variables to be zero. Since all observations in these variables had zero value, we need to remove them because we can't use them by using step_nzv() function. SVR, XGBoost, LightGBM, and RF parameters are tuned using tune race anova. Tune race anova is faster than grid search and random search. These four models were regressed with 5 repeated 10-folds cross validation and 7 differents data. The results are as follows:
![image](https://user-images.githubusercontent.com/48485276/197677603-ddd87ea4-4bf9-4f77-9f65-6f1f6c6d6201.png)
![image](https://user-images.githubusercontent.com/48485276/197678084-b5383906-5acb-4652-b7f8-e14dda372d0c.png)
The combination of SVM and batchnorm data gave us the lowest RMSE while the combination of Random Forest and and L1L2 data gave us the highest R-squared. The combination of Random Forest and and L1L2 were also the best overall model (highest R-squared and 4th lowest RMSE).

![image](https://user-images.githubusercontent.com/48485276/197679673-8bb6a486-9f4d-418c-98d5-6bd59cac9568.png)

#### Creating Predictions
We had two candidate models, SVM-batchnorm and RF-L1L2. Predictions will be made using these two candidates. The predictions is plotted using scatter plot with three additional lines, a solid line and two dashed lines. The solid line refers to the diagonal line indicating perfect predictions while the dashed line refers to prediction error (RMSE). Tha RMSE value was taken from the 5-repeated 10-folds cross-validation.

![image](https://user-images.githubusercontent.com/48485276/197680434-539c243d-a6f3-4f7f-9edb-be847281f579.png)
This was the result of SVM-batchnorm. The predictions were excellent with only one underestimate prediction.

![image](https://user-images.githubusercontent.com/48485276/197681722-24a86744-bf5b-4110-84c8-b43de9508c2c.png)
The result of RF-L1L2 predictions was terrible. The predictions didn't even show a straight line and seemed to create two clusters.

## What to pay attention to
I forgot to set a seed before training the image classification. tf.random.set_seed() can be used to set seed for reproducibility and put before Sequential(). We will update this later. 


