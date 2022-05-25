# DEMO

## TIPS:

In all the demo, controller got three inputs.

- First one determine module, now support 

  - [x] cigp

  - [x] hogp

  - [x] GAR

  - [x] CIGAR

  - [x] NAR

  - [x] MF-BNN

  - [x] DC

  - [x] ResGP

  - [x] AR

    

- Second one is controller config. Got the follow parameters. (check default value in [main_controller.py](../utils/main_controller.py))

  - [x] check_point(list): while epoch reach checkpoint, controller will eval at that epoch
  - [x] record_step(int): while epoch%record_step==0, controller save module parameters, when training meet nan error, try to reload record parameters.
  - [x] max_epoch(int): max training epoch

  

- Third on is module config. Got parameters depending on module type. The module has its own default config defined, most of then are the same. The parameters always contain the follows:

  - learning rate
  - kernel define 
  - input normalize
  - output normalize
  - param exp_restrict
  - noise init value



## Demo available

After running the demo, the result will be saved in ["../exp"](../exp) folder.

#### 1.GAR

```
python demo/normal_data/demo_multi-fidelity_hogp.py
```



#### 2.CIGAR

```
python demo/normal_data/demo_ind_multi_fidelity_hogp.py
```



#### 3.NAR

```
python demo/normal_data/demo_nar_cigp.py
```



#### 4.MF-BNN

```
python demo/normal_data/demo_dmfal.py
```



#### 5.DC

```
python demo/normal_data/demo_dc_cigp.py
```



#### 6.ResGP

```
python demo/interp_data/demo_res_gp.py
```



#### 7.AR

```
python demo/interp_data/demo_lar_gp.py
```

