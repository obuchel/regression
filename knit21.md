Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

Dataset

The training data for this project are available here: <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv> The test data are available here: <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv> The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>.

Data Preprocessing

Loading the R packages:

``` r
# packages
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(rattle)
```

    ## Rattle: A free graphical interface for data mining with R.
    ## Version 4.0.5 Copyright (c) 2006-2015 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

``` r
library(rpart)
library(rpart.plot)
library(randomForest)
```

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(repmis)
```

You can also embed plots, for example:

    ## [1] TRUE

    ##                      freqRatio percentUnique zeroVar   nzv
    ## roll_belt             1.101904     6.7781062   FALSE FALSE
    ## pitch_belt            1.036082     9.3772296   FALSE FALSE
    ## yaw_belt              1.058480     9.9734991   FALSE FALSE
    ## total_accel_belt      1.063160     0.1477933   FALSE FALSE
    ## gyros_belt_x          1.058651     0.7134849   FALSE FALSE
    ## gyros_belt_y          1.144000     0.3516461   FALSE FALSE
    ## gyros_belt_z          1.066214     0.8612782   FALSE FALSE
    ## accel_belt_x          1.055412     0.8357966   FALSE FALSE
    ## accel_belt_y          1.113725     0.7287738   FALSE FALSE
    ## accel_belt_z          1.078767     1.5237998   FALSE FALSE
    ## magnet_belt_x         1.090141     1.6664968   FALSE FALSE
    ## magnet_belt_y         1.099688     1.5187035   FALSE FALSE
    ## magnet_belt_z         1.006369     2.3290184   FALSE FALSE
    ## roll_arm             52.338462    13.5256345   FALSE FALSE
    ## pitch_arm            87.256410    15.7323412   FALSE FALSE
    ## yaw_arm              33.029126    14.6570176   FALSE FALSE
    ## total_accel_arm       1.024526     0.3363572   FALSE FALSE
    ## gyros_arm_x           1.015504     3.2769341   FALSE FALSE
    ## gyros_arm_y           1.454369     1.9162165   FALSE FALSE
    ## gyros_arm_z           1.110687     1.2638875   FALSE FALSE
    ## accel_arm_x           1.017341     3.9598410   FALSE FALSE
    ## accel_arm_y           1.140187     2.7367241   FALSE FALSE
    ## accel_arm_z           1.128000     4.0362858   FALSE FALSE
    ## magnet_arm_x          1.000000     6.8239731   FALSE FALSE
    ## magnet_arm_y          1.056818     4.4439914   FALSE FALSE
    ## magnet_arm_z          1.036364     6.4468454   FALSE FALSE
    ## roll_dumbbell         1.022388    84.2065029   FALSE FALSE
    ## pitch_dumbbell        2.277372    81.7449801   FALSE FALSE
    ## yaw_dumbbell          1.132231    83.4828254   FALSE FALSE
    ## total_accel_dumbbell  1.072634     0.2191418   FALSE FALSE
    ## gyros_dumbbell_x      1.003268     1.2282132   FALSE FALSE
    ## gyros_dumbbell_y      1.264957     1.4167771   FALSE FALSE
    ## gyros_dumbbell_z      1.060100     1.0498420   FALSE FALSE
    ## accel_dumbbell_x      1.018018     2.1659362   FALSE FALSE
    ## accel_dumbbell_y      1.053061     2.3748853   FALSE FALSE
    ## accel_dumbbell_z      1.133333     2.0894914   FALSE FALSE
    ## magnet_dumbbell_x     1.098266     5.7486495   FALSE FALSE
    ## magnet_dumbbell_y     1.197740     4.3012945   FALSE FALSE
    ## magnet_dumbbell_z     1.020833     3.4451126   FALSE FALSE
    ## roll_forearm         11.589286    11.0895933   FALSE FALSE
    ## pitch_forearm        65.983051    14.8557741   FALSE FALSE
    ## yaw_forearm          15.322835    10.1467740   FALSE FALSE
    ## total_accel_forearm   1.128928     0.3567424   FALSE FALSE
    ## gyros_forearm_x       1.059273     1.5187035   FALSE FALSE
    ## gyros_forearm_y       1.036554     3.7763735   FALSE FALSE
    ## gyros_forearm_z       1.122917     1.5645704   FALSE FALSE
    ## accel_forearm_x       1.126437     4.0464784   FALSE FALSE
    ## accel_forearm_y       1.059406     5.1116094   FALSE FALSE
    ## accel_forearm_z       1.006250     2.9558659   FALSE FALSE
    ## magnet_forearm_x      1.012346     7.7667924   FALSE FALSE
    ## magnet_forearm_y      1.246914     9.5403119   FALSE FALSE
    ## magnet_forearm_z      1.000000     8.5771073   FALSE FALSE
    ## classe                1.469581     0.0254816   FALSE FALSE

The cleaned data sets clearnTrainData and cleanTestData both have 53 columns with the same first 52 variables and the last variable classe and problem\_id individually. cleanTrainData has 19622 rows while cleanTestData has 20 rows.

Data spliting

I split the cleaned training set trainData into a training set (train, 70%) for prediction and a test set (test 30%).

``` r
set.seed(7826) 
inTrain <- createDataPartition(cleanTrainData$classe, p = 0.7, list = FALSE)
train <- cleanTrainData[inTrain, ]
test <- cleanTrainData[-inTrain, ]
```

Prediction Algorithms

First, the “out of the box” classification tree and random forests to predict outcomes for classe.

``` r
ctrl <- trainControl(method = "cv", number = 5)
modfit <- train(classe ~ ., data = train, method = "rpart", 
                   trControl = ctrl)
print(modfit, digits = 4)
```

    ## CART 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10989, 10989, 10990, 10989, 10991 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp       Accuracy  Kappa    Accuracy SD  Kappa SD
    ##   0.03723  0.5241    0.38748  0.03851      0.06202 
    ##   0.05954  0.4144    0.20668  0.06477      0.10984 
    ##   0.11423  0.3482    0.09762  0.03575      0.05469 
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was cp = 0.03723.

Accuracy was used to select the optimal model using the largest value. The final value used for the model was cp = 0.03723.

``` r
fancyRpartPlot(modfit$finalModel)
```

![](knit21_files/figure-markdown_github/unnamed-chunk-5-1.png)

``` r
# Predicting outcomes using test dataset
predict <- predict(modfit, test)
# Prediction results
(confidence <- confusionMatrix(test$classe, predict))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1544   21  107    0    2
    ##          B  492  391  256    0    0
    ##          C  474   38  514    0    0
    ##          D  436  175  353    0    0
    ##          E  155  138  293    0  496
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5004          
    ##                  95% CI : (0.4876, 0.5133)
    ##     No Information Rate : 0.5269          
    ##     P-Value [Acc > NIR] : 1               
    ##                                           
    ##                   Kappa : 0.3464          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.4979  0.51245  0.33749       NA  0.99598
    ## Specificity            0.9533  0.85396  0.88262   0.8362  0.89122
    ## Pos Pred Value         0.9223  0.34328  0.50097       NA  0.45841
    ## Neg Pred Value         0.6303  0.92162  0.79234       NA  0.99958
    ## Prevalence             0.5269  0.12965  0.25879   0.0000  0.08462
    ## Detection Rate         0.2624  0.06644  0.08734   0.0000  0.08428
    ## Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
    ## Balanced Accuracy      0.7256  0.68321  0.61006       NA  0.94360

``` r
##Verifying accuracy of prediction...
(accuracy <- confidence$overall[1])
```

    ##  Accuracy 
    ## 0.5004248

The accuracy rate of classification tree is 0.5 does not warrant good prediction. For thsi reason, we use random forests.

Random forests

``` r
random_forests <- train(classe ~ ., data = train, method = "rf", trControl = ctrl)
print(random_forests, digits = 4)
```

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10990, 10990, 10990, 10988, 10990 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy  Kappa   Accuracy SD  Kappa SD
    ##    2    0.9905    0.9880  0.002697     0.003414
    ##   27    0.9908    0.9884  0.003523     0.004459
    ##   52    0.9851    0.9811  0.004618     0.005847
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 27.

Accuracy was used to select the optimal model using the largest value. The final value used for the model was mtry = 27.

``` r
# Confusion matrix
predictRF <- predict(random_forests, test)
(confRF <- confusionMatrix(test$classe, predictRF))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1668    3    0    1    2
    ##          B   10 1125    3    1    0
    ##          C    0    2 1017    7    0
    ##          D    0    1   15  944    4
    ##          E    2    2    1    4 1073
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9901          
    ##                  95% CI : (0.9873, 0.9925)
    ##     No Information Rate : 0.2855          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9875          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9929   0.9929   0.9817   0.9864   0.9944
    ## Specificity            0.9986   0.9971   0.9981   0.9959   0.9981
    ## Pos Pred Value         0.9964   0.9877   0.9912   0.9793   0.9917
    ## Neg Pred Value         0.9972   0.9983   0.9961   0.9974   0.9988
    ## Prevalence             0.2855   0.1925   0.1760   0.1626   0.1833
    ## Detection Rate         0.2834   0.1912   0.1728   0.1604   0.1823
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9957   0.9950   0.9899   0.9912   0.9963

Verifying accuracy

``` r
# Verifying accuracy
(accuracyRF <- confRF$overall[1])
```

    ##  Accuracy 
    ## 0.9901444

Random forest method has a much higher accuracy rate is 0.991 than classifications trees, and so the out-of-sample error rate is 0.009. But the algorithm itself is difficult to interpret and is computationally inefficient. It took around 15 minutes to compute the results.

Prediction on Testing Set with Random Forests

``` r
# Run against 20 testing set provided by Professor Leek.
(predict(random_forests, cleanTestData))
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Appendix

Correlation Matrix

``` r
library(corrplot)
 M <- cor(cleanTestData)
 corrplot(M, method = "circle", order = "FPC")
```

![](knit21_files/figure-markdown_github/unnamed-chunk-12-1.png)
