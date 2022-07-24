TOC

[toc]



## ML_GP

machine learning gaussian process implement



## Environment setup

We recommend using anaconda to setup the python environment. The following are test base on x86 platform.

```
conda create -n ML_GP python=3.8
conda activate ML_GP
conda install numpy
conda install -c pytorch pytorch
conda install -c conda-forge tensorly
conda install scipy
conda install -c conda-forge scikit-learn
```



## Demo

[see here](./demo/README.md)





## Baisc module

### [GP_module(use CIGP module as example)](./module/cigp.py)

All module contains it's own default config. For CIGP, it's 

```yaml
default_module_config = {
    'dataset' : {'name': 'Piosson_mfGent_v5',
                 'interp_data': False,
                 
                 # preprocess
                 'seed': None,
                 'train_start_index': 0,
                 'train_sample': 8, 
                 'eval_start_index': 0, 
                 'eval_sample':256,
                
                 'inputs_format': ['x[0]'],
                 'outputs_format': ['y[2]'],

                 'force_2d': True,
                 'x_sample_to_last_dim': False,
                 'y_sample_to_last_dim': True,
                 'slice_param': [0.6, 0.4], #only available for dataset, which not seperate train and test before
                 },

    'lr': {'kernel':0.01, 
           'optional_param':0.01, 
           'noise':0.01},

    'kernel': {
            'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
              },
    'evaluate_method': ['mae', 'rmse', 'r2'],
    'optimizer': 'adam',
    'exp_restrict': False,
    'input_normalize': True,
    'output_normalize': True,
    'noise_init' : 1.,
    'cuda': False,
}
```

- for 'dataset' part, refer to [data_loader](###3.data_loader) and [data_preprocess](###4.data_preprocess).

- 'lr' is learning rate. Learning rate is define as dict. You can use it to set different param learning rate. (TODO support step learning rate)

- 'kernel' define what kernel is use. It also support multi-kernel. For multi-kernel, it can be define as 

  ```yaml
  'kernel': {
              'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
              'K2': {'Linear': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
                },
  ```

- 'evaluate_method 'define how to evaluate the inference result.

- 'optimizer'. Pytorch optimizer. TODO support SGD.
- 'exp_restrict'. If set true, then the noise will be in torch.exp(noise) format.
- 'noise_init'. noise initialize value. It should be float, not int.
- 'input_normalize'/'output_normalize'. If set true, then the data will be preprocess by normalization.
- 'cuda'. If true, using cuda to train/eval.



### [training controller](./utils/main_controller.py)

Training controller using to record result, save module training status, (TODO: restore last step when training meet nan error)

It also has default config as follow:

```yaml
default_controller_config = {
    'batch_size': 1, # unvalid now 
    'check_point': [1, 10, 100, 300, 500, 1000, 2500, 5000, 7500, 10000],
    'eval_batch_size': 1, # unvalid now 
    'record_step': 50, # unvalid now 
    'max_epoch': 1000,
}
```

- 'max_epoch'. Max training epoch
- check_point. When epoch-number is contained in 'check_point'. Controller will start evaluation and record it.

While runing, the training controller will generate and write train/eval information on file. The example case is [here](./record.txt). Actually it record almost everything that need to reproduce training result.

TODO - record git.commit_id.

TODO - record experment start date.



## Utils

### 1. [mlgp_log](./utils/mlgp_log.py)

​	In this repository, We using a special log module to print information clearly. We have four level method as follow:

- info. White font, Black background. 
- warning. Yellow font, Black background. 
- error. White font, Red background.
- debug. White font, Purple background.

​	For example to see how it work, just run 

```
python utils/mlgp_log.py
```



### 2.[mlgp_record](./utils/mlgp_result_record.py)

​	This is a simply result record function. Using to record experiment result in a determine format.

- When initialize recorder module, you can assign append info (It can be list/dictionary/value/string). The append info will be record in the file between a keyword - '@append_info@'. If a dictionary is given, the info will be write in *ymal* format.
- After initialization, call 'register' to record key words. Key word should be a list of strings. 

- After register, call 'record' to record the result. Result is supposed to be list/dictionary of values/strings. If using list, then the order in list is important. If using dictionary, auto-reorder will be applied. This line will write in 'csv' format, which is easy to check using 'office.excel'

​	For example to see how it work, just run

```
python utils/mlgp_result_record.py
```

​	After running this code, a file name 'record_test.txt' will be generate. And the result is like this

```
@MLGP_recorder@
@append_info@
Function: A
Purpose: Test
time:
  now: 2022-07-17 19:57:49.001955
  weekdata: 7
@append_info@
epoch,result
0,0.5
1,0.6
2,0.7
```

TODO - load and parse record.



### 3.[data_loader](./utils/data_utils/data_loader.py)

​	Due to various formats on different dataset, data_loader only require 4 returning results. It's determined to be order like this "x_train, y_train, x_eval, y_eval". The 'x_eval, y_eval' can be None. But 'x_train, y_train' is need. 

​	The default support format is '.mat', format of *matlab*. If you want to added new format support, the only requirement is returning 4 result.

 

### 4.[data_preprocess](./utils/data_utils/data_preprocess.py)

​	This data process has a default config and require 4 inputs. "x_train, y_train, x_eval, y_eval". The 'x_eval, y_eval' can be None. But 'x_train, y_train' is need. 

​	If a config is given when initializing

​	Default config as follow, it no config given when initializing, akk:

```yaml
preprocess_default_config_dict = {
    'seed': None,

    # now only available for dataset, which not seperate train and test before
    'slice_param': [0.6, 0.4],

    # define sample select
    'train_start_index': 0, 
    'train_sample': 8, 
    'eval_start_index': 0, 
    'eval_sample':256,
    
    # others
    'force_2d': False,
    'x_sample_to_last_dim': False,
    'y_sample_to_last_dim': False,
    
    # define multi fidelity input/output format
    'inputs_format': ['x[0]'],
    'outputs_format': ['y[0]', 'y[2]'],

}
```

- seed use to shuffle on x_train, y_train. Notice: shuffle only work on x_train,y_train, not work on x_eval, y_eval.

- slice_param. Only valid when x_eval, y_eval is None. Slice x_train, y_train into two parts, the first one is train_sample, the second one is eval sample. 

- train_start_index + train_sample. Meaning the output x_tr = x_tr[train_start_index: train_start_index+train_sample]

- eval_start_index + eval_sample. Work as train_start_index + train_sample on eval data.

- force_2d. If true, then the x, y will be reshape to 2D.

- x_sample_to_last_dim/y_sample_to_last_dim. It supposed that the x first dim determine sample. If true, put the first dim to last dim.

- inputs_format/ outputs_format. This is the last step of preprocess. It work as "eval(cmd)". Meaning that the strings in the list is supposed to be 'code'(for example, if x = a+b, then the code is 'a+b'). And the list means multi input/output. We use code to explain how it works.

  ```
  #Here, after the previous step, we got x, y.(actually got two pairs, x_tr, y_tr; x_te, y_te. This function will process data in the sample way for this two pairs). All is list. 
  
  #Suppose if we have input x0, dims like (n,m). y1(n,d1), y2(n,d2), y3(n,d3) as multi-fidelity.
  #Then we have
  x = [x0]
  y = [y1,y2,y3]
  
  #If model only need y1.
  'outputs_format': ['y[0]'],
  
  #if model need y2 - y1 as output
  'outputs_format': ['y[1] - y[0]'],
  
  #if model need y1, y3(in order)
  'outputs_format': ['y[0]', 'y[3]'],
  
  # if model need y3, y1(in order)
  'outputs_format': ['y[3]', 'y[0]'],
  
  #if model only need the first 2 value of dim-m of x.
  'inputs_format': ['x[0][:, 0:2]'],
  
  #if model only need the last 3 value of dim-m of x.
  'inputs_format': ['x[0][:, -2:]'],
  ```

  

### 5.[auto_nan_check(not full implement, need error case to check this function)](./utils/mlgp_hook)

​	TODO. need error case to test and optimize the code.



### 6.Record error status on determine function

​	Please refer here [how_to_apply_utils_on_your_code\README.md](how_to_apply_utils_on_your_code\README.md).



### 7.plot result.

​	Parse the mlgp_record.txt file and plot. [plot.py](plot.py)

​	This demo only show how to get data fomr mlgp_record.txt file. The ploting using the most simple method.