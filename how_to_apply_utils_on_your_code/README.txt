# How to apply some utils on your code

Here we take an example from https://github.com/AMMLE/EasyGP

For the following part, we use 'cigp_v10.py' as example.

And the optimize code was saved with name 'cigp_v10_refine.py'



## 1.Use MLGP_Recorder

From example we want to record iter,nll for every 10 training iterations,

```
# First import function at the head of file
from utils.mlgp_result_record import MLGP_recorder
```



```
# Init recorder. Actually we recommand init recorder outside model, but not inside model, due to the cigp code structure, the easy way is add it inside model

# The append info will be add to the record. Actually we should record all the information that leads to 
# reproduct the training result as far as we can.  
(Line64) append_info = {'module': str(self),'X_normalize':True, 'Y_normalize': True} 

# In this function, overlap was True to allow multi-record on single txt file
(Line65) self.rc = MLGP_recorder('./how_to_apply_utils_on_your_code/record.txt', append_info, overlap=True)

# register what result your want to save, here we use 'iter' and 'nll'
(Line66) self.rc.register(['iter', 'nll'])
```



```
# record while training
(Line 174) if (i+1)%10 == 0:
(Line 175)     self.rc.record([i, loss.item()])
```

Then after running result was save in './how_to_apply_utils_on_your_code/record.txt'. The append_info was saved based on yaml format(easy for human/machine to read). And the result was saved based on csv format.



## 2.Catch singular error with torch.cholesky

In some case, torch.cholesky or other function may meet uncaught error and stop the training.

Here we provide a function to catch the error status.

```
#import function
from utils.mlgp_hook import set_function_as_module_to_catch_error
```



```
# set torch.cholesky as function
(Line 70) self.torch_cholesky = set_function_as_module_to_catch_error(torch.cholesky)

# optional, set 'mlgp_record_file' path and then the function can write in with the error information.
(Line 71)os.environ['mlgp_record_file'] = './how_to_apply_utils_on_your_code/record.txt'
```



```
# using self.torch_cholesky to compute instead of torch.cholesky
(Line 125) # L = torch.cholesky(Sigma) 
(Line 126) L = self.torch_cholesky(Sigma)
```

```
(Line 154) # L = torch.cholesky(Sigma) 
(Line 155) L = self.torch_cholesky(Sigma)
```

Then, if running with any error in this function. The inputs, error reason will all be recorded. 



To get a singular error, I change the training code

```
(Line 149) # Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(
(Line 150) #    	  	self.X.size(0)) + JITTER * torch.eye(self.X.size(0))
```

was change to

```
(Line 151) Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1)* torch.eye(
(Line 152)  		   self.X.size(0))
```

Then the training will got an singular problem and the error log was save in [../exception_log/log_0](../exception_log/log_0). All the function inputs/error log will be saved. And to reproduce the result, put [./utils/tools_script/run_exception_case.py](utils/tools_script/run_exception_case.py) in the log folder and then run it.