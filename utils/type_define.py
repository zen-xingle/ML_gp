
class GP_val_with_bar:
    def __init__(self, mean, var) -> None:
        self.mean = mean
        self.var = var

    def get_mean(self):
        return self.mean
    
    def get_var(self):
        return self.var