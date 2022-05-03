

## Kernel类说明

```
Class kernel(torch.nn.Module):
#继承torch.nn.Module,便于反向传播
	def __init__(self)
	#有初始需求时，加输入参数
		...
	def forward(self,x1,x2)
	#目前暂规定kernel的输入为x1,x2
		...
	def get_param(self)
	#返回可训练参数
```



## 参考文档

doc refer: https://www.cs.toronto.edu/~duvenaud/cookbook/



## kernel list

- [x] Squared Exponential Kernel
- [x] Rational Quadratic Kernel
- [ ] Periodic Kernel
- [ ] Locally Periodic Kernel
- [x] Linear Kernel
- [x] combine Kernel
