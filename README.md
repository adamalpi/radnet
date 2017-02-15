# radnet
Deep neural network for climate science radiation functions

## Train
To train a model use train.py
Example:

```
python ./Code/ClimateSc/radnet/train.py --data_dir ./Data/radiation_data_v2/
```

In this case, we pass the data using the option "--data_dir". inside data are many folders each one containing samples of the dataset, so no matter how many nested dirs, the reader will reach the .csv files.
The model will generate a dir in which it places the models and the summaries of each model. The model will store only the latest models.

## Generate
To generate data 

```
python Code/ClimateSc/radnet/generate.py --data_dir ./Data/radiation_data_v2/data_19/ logdir/train/2017-02-13T14-34-09/model.ckpt-92000
```

Select the data for what you want to predict results from and the model to use.
The execution will create a climate_results dir in which the results will be stored in a pseudo .csv file where the first line is the mse of the sample and the rest lines follow the format \<original_result\>,\<predicted_result\>

## Helpers
helpers/stats_var_calculator.py calculates std and mean used in the model for normalizing the data.
helpers/graphics.ipynb helps to visualize the results from the climate_results folder.

## TensorBoard
Just pass the directory that contains the models and log summaries.

```
tensorboard --logdir ./logdir/
```

## Project features
The model is able to save the model, load a model and keep training it even with other hyperparameters, log loss summaries tensorboard and tune other features. For more help: 
