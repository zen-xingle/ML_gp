# ML-GP

machine learning gaussian process implement



## Environment Setup

<details>
<summary>&#160;&#160;&#160; Based on anaconda</summary>
<pre style="background-color:WhiteSmoke;">
conda create -n ML_GP python=3.8
conda activate ML_GP
conda install numpy
conda install -c pytorch pytorch
conda install -c conda-forge tensorly
conda install scipy
conda install -c conda-forge scikit-learn
</pre>
</details>
<br>


## Quick Start
The GP modules support directly call to perform a demo based on some generated data and visualize the result as follow:
```
python gp_module/CIGAR.py
python gp_module/GAR.py
python gp_module/DC_I.py
python nn_net/MF_BNN.py
```
These demo show you how to feed the data and get the predict result.
<br> 

## Demo

[see here](./demo/README.md)
<br>

##Appendix
<details>
<summary>&#160;Hyperparameters - GP module</summary>
&#160;&#160;&#160;Almost all the hyperparameters can be set via a Python "Dictionary". Defining in this way can help us store the hyperparameters, which is convenient for reproducing results.<br>
&#160;&#160;&#160;Take CIGP module as an example. The hyperparameters are defined in the head of <a href="./gp_module/cigp.py">cigp.py</a>.<br>

<pre style="background-color:WhiteSmoke;">
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
    'weight_decay': 1e-3,

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
</pre>
&#160;&#160;&bull;&#160; 'dataset': record hyperparam of dataset.
&#160;&#160;&bull;&#160; 'lr': : Learning rate. Learning rate is defined as dict. You can use it to set different params with their specific learning rate.
&#160;&#160;&bull;&#160; 'weight_decay': Params penalty coefficient. Set higher to avoid overfitting.
&#160;&#160;&bull;&#160; 'kernel': Define what kernel is use. It also support multi-kernel. For multi-kernel, it can be define as
<pre style="background-color:WhiteSmoke;">
'kernel': {
            'K1': {'SE': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
            'K2': {'Linear': {'exp_restrict':True, 'length_scale':1., 'scale': 1.}},
              },
</pre>
&#160;&#160;&bull;&#160; 'evaluate_method': Define how to evaluate the inference result.
&#160;&#160;&bull;&#160; 'optimizer': Pytorch optimizer.
&#160;&#160;&bull;&#160; 'exp_restrict': If set true, the noise will be compute in torch.exp(noise).
&#160;&#160;&bull;&#160; 'input_normalize': If set true, the intput data will be preprocess by normalization.
&#160;&#160;&bull;&#160; 'output_normalize': If set true, the output data will be preprocess by normalization.
&#160;&#160;&bull;&#160; 'noise_init': Noise initialize value. It should be float, not int.
&#160;&#160;&bull;&#160; 'cuda': Set True to enable gpu acclerating.
</details>
<br>

<details>
<summary>&#160;Training setting</summary>
&#160;&#160;&#160;Training controller using to record result, save module training status. The setting was defined in the head of <a href="utils/main_controller.py">main_controller</a> as shown as followed.
<pre style="background-color:WhiteSmoke;">
default_controller_config = {
    'check_point': [1, 10, 100, 300, 500, 1000],
    'max_epoch': 1000,
    'record_file_dir': './exp/'}
</pre>
&#160;&#160;&bull;&#160; 'check_point': when reach check_point epoch, eval the module<br>
&#160;&#160;&bull;&#160; 'max_epoch': max training epoch<br>
&#160;&#160;&bull;&#160; 'record_file_dir': path to save result
</details>
<br>

<details>
<summary>&#160;Recorder and parser</summary>
&#160;&#160;&#160; Recorder and parser was define in <a href="./utils/mlgp_result_record.py">mlgp_result_record</a>. It mainly records the module hyperparameters and result.
Directly call it and it will generate "./record_test.txt" file, showing how recorder and parser work.
<pre style="background-color:WhiteSmoke;">
python utils/mlgp_result_record.py
</pre>
</details>
<br>

<details>
<summary>&#160;Data loader</summary>
&#160;&#160;&#160;Due to various formats on different dataset, data_loader only require 4 returning results. It's determined to be order like this "x_train, y_train, x_eval, y_eval". The 'x_eval, y_eval' can be None. But 'x_train, y_train' is need. All of them should be list.
<br>
<br>
&#160;&#160;&#160;The default support format is '.mat', format of matlab. If you want to added new format support, the only requirement is returning 4 result.
</details>
<br> 


<details>
<summary>&#160;Data preprocess</summary>
&#160;&#160;&#160;This data process has a default config and require 4 inputs. "x_train, y_train, x_eval, y_eval". The 'x_eval, y_eval' can be None. But 'x_train, y_train' is need.<br>
&#160;&#160;&#160;Take default config as example. Here we explain how it work:

<pre style="background-color:WhiteSmoke;">
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
    'outputs_format': ['y[0]', 'y[2]']}
</pre>
&#160;&#160;&bull;&#160; 'seed': Seed use for shuffling on x_train, y_train. Notice: shuffling only work on x_train,y_train, not work on x_eval, y_eval.
&#160;&#160;&bull;&#160; 'slice_param': Only valid when x_eval, y_eval is None. Slice x_train, y_train into two parts, the first one is train_sample, the second one is eval sample.
&#160;&#160;&bull;&#160; 'train_start_index', 'train_sample': Meaning the output x_tr = x_tr[train_start_index: train_start_index+train_sample].
&#160;&#160;&bull;&#160; 'eval_start_index', 'eval_sample': Meaning the output x_eval = x_eval[eval_start_index: eval_start_index+eval_sample].
&#160;&#160;&bull;&#160; 'force_2d': If true, then the x, y will be reshape to 2D.
&#160;&#160;&bull;&#160; 'x_sample_to_last_dim', 'y_sample_to_last_dim': It's supposed that the first dim determine sample. If true, put the first dim to last dim.
&#160;&#160;&bull;&#160; 'inputs_format', 'outputs_format': This is the last step of preprocess. It work as "eval(cmd)". Meaning that the strings in the list is supposed to be 'code'(for example, if x = a+b, then the code is 'a+b'). And the list means multi input/output. We use more code to explain how it works. This may be difficult to understand at first, but it work well for tracking how the preprocess was made on specific dataset. 

<pre style="background-color:WhiteSmoke;">
#Here, after the previous step, we got x, y.(actually got two pairs, x_tr, y_tr; x_te, y_te. This function will process data in the sample way for this two pairs). All of them are list. 

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
</pre>
</details>
<br> 


<details>
<summary>&#160;Plot result</summary>
Try and see how it work:
<pre style="background-color:WhiteSmoke;">
python visualize_tools/plot.py
</pre>
</details>
<br>


<details>
<summary>&#160;Plot 2D field</summary>
Try any quick-start demo and see how it work after the demo finish training:
<pre style="background-color:WhiteSmoke;">
python gp_module/CIGAR.py
</pre>
</details>
<br>

