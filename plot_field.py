import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


class plot_container:
    def __init__(self, data_list, label_list, sample_dim) -> None:
        # data_list: containing data want to plot. suppose to be 3D(2D field + sample dim)
        # label_list: label attach to title
        # sample dim: set which dim is sample dim

        # check 2D, now support 2D plot
        shape=None
        for i,_d in enumerate(data_list):
            assert len(_d.shape)==3, "now support 2D field, but got {} for data-{}".format(_d.shape, i)
            if shape is not None:
                assert shape == _d.shape, "shape should be all same as {}".format(shape)
            else:
                shape = _d.shape

        self.data_list = data_list
        self.label_list = label_list
        self.sample_dim = sample_dim

        self.sample_number = data_list[0].shape[sample_dim]
        self.data_number = len(data_list)
        self.dims = data_list[0].shape

        self.field_dims = [i for i in range(len(data_list[0].shape))]
        self.field_dims.remove(sample_dim)
        self.field_dims = tuple(self.field_dims)

        #['flat', 'nearest', 'gouraud', 'auto', 'jet']
        self.map_type = 'jet'
        #['flat', 'nearest', 'gouraud', 'auto']
        self.shading_method = 'gouraud'

        self.get_min_max()


    def get_min_max(self):
        min_list = [_d.min(self.field_dims) for _d in self.data_list]
        max_list = [_d.max(self.field_dims) for _d in self.data_list]

        self.min_list = np.vstack(min_list).min(0)
        self.max_list = np.vstack(max_list).max(0)

        self.min_max_base_on_all_sample = False
        # if false, min_max get from everysample

    
    def plot(self):


        for j in range(self.sample_number):
            fig, axs = plt.subplots(nrows=1, ncols=self.data_number, figsize = (10, 5.5))
            # 一次只能画一个平面图，多个样本依次绘出    

            for i in range(self.data_number):
                _d = self.data_list[i]
                _l = self.label_list[i]
                ax = axs[i]

                get_sample_cmd = [':']* (len(self.dims)-1)
                get_sample_cmd.insert(self.sample_dim, 'j')
                get_sample_cmd = '_d[{}]'.format(','.join(get_sample_cmd))
                _d = eval(get_sample_cmd)
                
                if self.min_max_base_on_all_sample is True:
                    pcm = ax.pcolormesh(_d, cmap=self.map_type, shading=self.shading_method, vmin=self.min_list.min(), vmax=self.max_list.max())
                else:
                    pcm = ax.pcolormesh(_d, cmap=self.map_type, shading=self.shading_method, vmin=self.min_list[i], vmax=self.max_list[i])
                ax.tick_params(labelsize = 8)
                ax.set_title(str(self.label_list[i]), fontsize = 8)

            fig.tight_layout()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            plt.colorbar(pcm, cax=cax)
            # plt.rcParams.update({'font.size': 30})
            plt.tight_layout()
            plt.show()
            # input("AnyKey to continue")
            # plt.savefig(r'fig_new/' + f + '_' + str(sample_index) +'.eps', bbox_inches = 'tight')



if __name__ == '__main__':
    a = np.load('groundtrue.npy')
    b = np.load('result.npy')
    pc = plot_container([a,b,abs(a-b)], ['groundtrue', 'gar', 'diff'], 2)
    pc.plot()
