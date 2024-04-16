class model()
    def __init__(self):
        # init code
    def fit(self,X,y,lr,val_rate,batch_size, epoch)
        #....
    def get_parameters(self):
        return parameters #tensor
    def load_parameters(self, params)
        # upadate params to model
    def evaluate(self,X_test,y_test):
        return accuracy