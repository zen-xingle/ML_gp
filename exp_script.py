import torch


# dataset的形式
# A.配置
# B.自己生成
data_config = {
}

gmb = GP_model_block(
    dnm = '',
    gp_model = '',
    pre_process = '',
    post_process = '',
)

for iter in range(max_iters):
    opt = torch.optimize.adam(gmb.get_train_params())
    loss = gmb.train(dataset.get_train_data())
    opt.backward()

gmb.val(dataset.get_eval_data())

gmb_high_f = GP_model_block(
    dnm = '',
    gp_model = '',
    pre_process = '',
    post_process = '',
)
inputs, outputs = dataset_2.get_train_data()
inputs, outputs = gmb.forward_with_detecing_subset(inputs, outputs)
for iter in range(max_iters):
    opt = torch.optimize.adam(gmb_high_f.get_train_params())
    loss = gmb_high_f.train(inputs, outputs)
    opt.backward()




