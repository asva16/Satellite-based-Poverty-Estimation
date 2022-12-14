# Satellite-based-Poverty-Estimation
Estimating Expenditure-based Poverty using Satellite Images

## Introduction
Poverty is a global-scale problem that concerns all countries, including Indonesia. Poverty, in a socio-economic context, is defined as the inability of a person of a household to meet their primary needs, i.e., clothing, shelter, and food. United Nations is committed to eradicating poverty by 2030, as stated in the first goals of Sustainable Development Goals (SDGs). Relevant data is needed to implement policies related to poverty eradication. There are five approaches to collecting poverty data: the financial approach, basic needs approach, capabilities approach, social exclusion approach, and rights approach. Most countries use household income or consumption as the basis for determining the welfare of the poor. Unfortunately, timely and reliable data collection is difficult and expensive because it requires detailed data collection directly from individuals and households.

New potential data sources have emerged to estimate poverty, such as mobile phone data or satellite imagery. Recent developments in machine learning-based methods have enabled an intensive approach to measuring various poverty indicators. Satellite images come in two forms, daytime satellite images and nighttime images. Daytime satellite images have a higher image resolution than nighttime satellite images. They provide plenty of helpful information in the form of colours and patterns. The approach taken in this work is replicated and modified from Jean's study (2016). This approach uses the Convolutional Neural Network (CNN) to determine the relationship between day and night satellite imagery. The CNN architecture was retrained (fine-tuned) to predict the intensity of night light according to the input of daytime satellite imagery.

Using the Convolutional Neural Network (CNN) and different regression models, we wish to estimate the per capita expenditure of 40 regencies in Central Java and the Special Region of Yogyakarta. While fine-tuning VGG16, we also reduced the last fully connected layer to 32 nodes to avoid the curse of dimensionality. Since the data extracted from satellite images are not interpretable, we use Support Vector Regression (SVR), Extreme Gradient Boosting (XGBoost), and Random Forest. Out of the five poverty measurement approaches, the consumption expenditure-based monetary approach is the most appropriate response variable because only the residence dimension can be captured by satellite imagery. By using only public, accessible, and quickly updated data, the results of this study are expected to provide a grim picture but provide up-to-date information in mapping the level of per capita expenditure.

## Result
### Classifying Images
![image](https://user-images.githubusercontent.com/48485276/197674253-53cbb4cb-f96f-4e2b-b94c-472a2899bd0c.png)
We tried a different method to lower the validation loss, as shown in the picture above. Even though applying dropout produced a model with the lowest validation loss, the model didn't give a convergent result. A model that did make a convergent result was a model with augmentation. It had almost the same minimum validation loss as the dropout model. But with more research and time, it's possible to lower the validation loss even further.

![image](https://user-images.githubusercontent.com/48485276/197675251-62ee9031-4691-4da0-b304-d386ab854864.png)
As for accuracy, the dropout model, augment model, and base model gave us almost a convergence result. Looking at the graph, the augment model is still the best one.

We expected that the data generated from augment model are good predictors to estimate household expenditure.

### Regression
#### Modeling
Using the relu activation function caused some variables to be zero. Since all observations in these variables had zero value, we need to remove them because we can't use them by using step_nzv() function. SVR, XGBoost, LightGBM, and RF parameters are tuned using tune race anova. Tune race anova is faster than grid search and random search. These four models were regressed with 5-repeated 10-folds cross-validation and 7 different data. The results are as follows:

![image](https://user-images.githubusercontent.com/48485276/197677603-ddd87ea4-4bf9-4f77-9f65-6f1f6c6d6201.png)
![image](https://user-images.githubusercontent.com/48485276/197678084-b5383906-5acb-4652-b7f8-e14dda372d0c.png)
The combination of SVM and batchnorm data gave us the lowest RMSE, while the combination of Random Forest and L1L2 data gave us the highest R-squared. The combination of Random Forest and L1L2 were also the best overall model (highest R-squared and 4th lowest RMSE). 

![image](https://user-images.githubusercontent.com/48485276/197679673-8bb6a486-9f4d-418c-98d5-6bd59cac9568.png)

#### Creating Predictions
We had two candidate models, SVM-batchnorm and RF-L1L2. Predictions will be made using these two candidates. The predictions are plotted using a scatter plot with three additional lines, a solid line and two dashed lines. The solid line indicates the diagonal line marking perfect predictions, while the dashed line refers to prediction error (RMSE). The RMSE value was obtained from the 5-repeated 10-folds cross-validation.

![image](https://user-images.githubusercontent.com/48485276/197680434-539c243d-a6f3-4f7f-9edb-be847281f579.png)

This was the result of SVM-batchnorm. The predictions were excellent, with only one underestimate prediction.

![image](https://user-images.githubusercontent.com/48485276/197681722-24a86744-bf5b-4110-84c8-b43de9508c2c.png)

The result of RF-L1L2 predictions was terrible. The predictions didn't even show a straight line and seemed to create two clusters.

[Updated]

![image](https://user-images.githubusercontent.com/48485276/197712004-8aac386a-0a2e-4b73-937b-697f48145dea.png)

This was the result of SVM-Augment. The predictions weren't better than SVM-batchnorm. SVM-Augment produces 3 underestimate predictions. From these two figures alone, we expected that SVM-batchnorm was the best model.

## Conclusions and Recommendations
It is certainly possible to predict poverty using satellite images. This work only describes the steps in a simple way. Another way is by using Image Segmentation to extract number of houses, the length of roads, and etc. More regression observations are preferable as this can generate more stable result. Satellite-based poverty estimation works better in low-level administration such as villages as the coverage are of an village is quite small. 

## What to pay attention to
1.  We forgot to set seed before training the image classification. tf.random.set_seed() can be used to set seed for reproducibility and is put before Sequential(). We will set seed for the next CNN project(s).
2.  Both batchnorm and L1l2 data were generated from non-convergent CNN model. It seems that we don't even need a good CNN model to extract features from satellite images. We will add a model that is trained with augment data later and compare the regression result. [Updated]
3.  Since the best regression model used batchnorm CNN model to extract image features, we couldn't figure out the relationship between CNN model and the regression model. Is plain transfer learning is enough to generate good regression features? Should we construct a CNN model that has the lowest possible loss (to extract image features) or just build the best regression model with any given data?

Credit to : https://github.com/carsonluuu/Poverty-Prediction-by-Satellite-Imagery
