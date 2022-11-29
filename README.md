Building a neural network for MNIST data
================
Chad Allison \| 29 November 2022

------------------------------------------------------------------------

Here, I will build a neural network model to identify handwritten digits
using the [MNIST
dataset](https://www.kaggle.com/c/digit-recognizer/overview) which I
have obtained from Kaggle.

------------------------------------------------------------------------

### Importing libraries and setting some quality of life options

``` r
library(tidyverse)
library(keras)
library(tensorflow)
library(yardstick)
library(tidymodels)

knitr::opts_chunk$set(message = F, warning = F)
options(scipen = 999)
theme_set(theme_classic())
```

------------------------------------------------------------------------

### Importing the data

``` r
mnist = read_csv("mnist_data.csv", col_types = cols())
head(mnist)
```

    ## # A tibble: 6 x 785
    ##   label pixel0 pixel1 pixel2 pixel3 pixel4 pixel5 pixel6 pixel7 pixel8 pixel9
    ##   <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
    ## 1     1      0      0      0      0      0      0      0      0      0      0
    ## 2     0      0      0      0      0      0      0      0      0      0      0
    ## 3     1      0      0      0      0      0      0      0      0      0      0
    ## 4     4      0      0      0      0      0      0      0      0      0      0
    ## 5     0      0      0      0      0      0      0      0      0      0      0
    ## 6     0      0      0      0      0      0      0      0      0      0      0
    ## # ... with 774 more variables: pixel10 <dbl>, pixel11 <dbl>, pixel12 <dbl>,
    ## #   pixel13 <dbl>, pixel14 <dbl>, pixel15 <dbl>, pixel16 <dbl>, pixel17 <dbl>,
    ## #   pixel18 <dbl>, pixel19 <dbl>, pixel20 <dbl>, pixel21 <dbl>, pixel22 <dbl>,
    ## #   pixel23 <dbl>, pixel24 <dbl>, pixel25 <dbl>, pixel26 <dbl>, pixel27 <dbl>,
    ## #   pixel28 <dbl>, pixel29 <dbl>, pixel30 <dbl>, pixel31 <dbl>, pixel32 <dbl>,
    ## #   pixel33 <dbl>, pixel34 <dbl>, pixel35 <dbl>, pixel36 <dbl>, pixel37 <dbl>,
    ## #   pixel38 <dbl>, pixel39 <dbl>, pixel40 <dbl>, pixel41 <dbl>, ...

The data has 42,000 observations with 785 variables. The first variable,
`label` is our outcome variable, while the rest of the variables are the
brightness of the pixel indicated by the variable name.

------------------------------------------------------------------------

### Cross validation

We will split the data into training and testing data, electing to use
80% as training data and the remaining 20% as testing data. We will also
identify `label` as our outcome variable in order to preserve balance
between classes.

``` r
split = initial_split(mnist, prop = 0.8, strata = "label")
train = training(split)
test = testing(split)

# creating a `set` variable to combine the two and create visual
train$set = "train"
test$set = "test"

rbind(train, test) |>
  mutate(label = factor(label)) |>
  group_by(set) |>
  count(label) |>
  mutate(prop = ifelse(set == "train", round(n / nrow(train), 3), round(n / nrow(test), 3))) |>
  ggplot(aes(label, prop)) +
  geom_col(aes(fill = set), position = "dodge") +
  scale_fill_manual(values = c("#CAA2D0", "#90BD90")) +
  labs(title = "class balance between testing and training data") +
  theme(plot.title = element_text(hjust = 0.5))
```

![](mnist-nnet_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
# removing the `set` variable for later use
train = train |> select(-set)
test = test |> select(-set)
```
