import torch
from CGan import CGAN
from model import Net
import argparse


class Argument():
    def __init__(self,data):
        self.noise_size=5
        self.n_features=28
        self.n_classes=8
        self.epoch=10
        self.batch_size=32
        self.dataset=data #tensor
        self.lrG=0.003
        self.lrD=0.003 

xo
parser = argparse.ArgumentParser(description='apply for federated learning framework')
parser.add_argument('--noise_size', type=int, default=100, help='Size of the noise vector')
parser.add_argument('--n_features', type=int, default=28*28, help='Number of features for the generator')
parser.add_argument('--n_classes', type=int, default=10, help='Number of classes in the dataset')
parser.add_argument('--epoch', type=int, default=10, help='Number of local epoch' )
parser.add_argument('--batch_size', type=int, default=32, help='Batch size' )
parser.add_argument('--lrG', type=float, default=0.001, help='learning rate of generator network' )
parser.add_argument('--lrD', type=float, default=0.001, help='learning rate of discriminator network' )

args = parser.parse_args()





class client():
    def __init__(self,args,data):
       self.cid=args.cid
       self.trainset=data #tensor
       self.testset=tes #tensor
       self.labels = torch.unique(self.trainset[:,:1].squeeze())
       args=Argument(self.trainset) #customs

       self.Gan=CGAN(args)
       self.model=Net(0.00)
       
    def update_model(self,params):
        self.model.load_parameters(params)
    def update_Gan(self,params):
        self.Gan.load_parameters(params)
    def Gan_fit(self,data):
        self.Gan.train(data)
        #code Generator fit data
    def model_fit(self,data):
        if len(data)!=0:
            X=data[:,1:].float()
            y=data[:,:1]
            y=y.squeeze().tolist()
            y=torch.tensor(y).long()
        else:
            X=self.trainset[:,1:].float()
            y=self.trainset[:,:1]
            y=y.squeeze().tolist()
            y=torch.tensor(y).long()
        self.model.fit(X,y,0.003,0.2,10)
        #model classify fit
    def get_parameters(self):
        D_parameters,G_parameters=self.Gan.get_parameters()
        M_parameters=self.model.get_parameters()
        return M_parameters,D_parameters, G_parameters
    def evaluate(self):
        X=self.testset[:,1:].float()
        y=self.testset[:,:1]
        y=y.squeeze().tolist()
        y=torch.tensor(y).ft()
        accuracy=self.model.evaluate(X,y)
        return accuracy
    

def fn_client(cid)->client:
    return client(cid)


class server():
    def __init__(self):
       self.trainset=torch.tensor(server_data.values) #tensor
       self.testset=torch.tensor(test_data.values) #tensor
       args=Argument(self.trainset) #customs

       self.Gan=CGAN(args)
       self.model=Net(0.00)
       
    def update_model(self,params):
        self.model.load_parameters(params)
    def update_Gan(self,params):
        self.Gan.load_parameters(params)
    def Gan_freedata_fit(self):
        self.Gan.freedata_train(self.model)
        #code Generator fit data
    def model_fit(self,data):
        if len(data)!=0:
            X=data[:,1:].float()
            y=data[:,:1]
            y=y.squeeze().tolist()
            y=torch.tensor(y).long()
        else:
            X=self.trainset[:,1:].float()
            y=self.trainset[:,:1]
            y=y.squeeze().tolist()
            y=torch.tensor(y).long()
        self.model.fit(X,y,0.003,0.2,10)
    def get_parameters(self):
        D_parameters,G_parameters=self.Gan.get_parameters()
        M_parameters=self.model.get_parameters()
        return M_parameters,D_parameters, G_parameters
    def evaluate(self):
        X=self.testset[:,1:].float()
        y=self.testset[:,:1]
        y=y.squeeze().tolist()
        y=torch.tensor(y).long()
        accuracy=self.model.evaluate(X,y)
        return accuracy
    def Gen_fake(self,n_samples):
        y=torch.randint(0, 8, (n_samples,))
        y=y.squeeze()
        return self.Gan.sample(y,n_samples)



class Federated_Learning():
    def __init__(self):
        self.server=server()
        self.testset=torch.tensor(test_data.values)
        self.clients=[]
        self.n_clients=8
        for i in range(self.n_clients):
            cl=fn_client(i)
            self.clients.append(cl)
    def client_M_update(self):
        M_params=self.server.model.get_parameters()
        for i in range(self.n_clients):
            self.clients[i].update_model(M_params)
    def server_M_update(self):
        params=self.clients[0].model.get_parameters()
        for i in range(len(params)):
            for k in range(1,self.n_clients):
                params[i]=params[i]+self.clients[k].model.get_parameters()[i]
            params[i]=params[i]/self.n_clients
        self.server.update_model(params)
        print("finished AVG model")

        
    def free_data_simulation(self,rounds):
        accuracy_hist=[]
        print(f"initial setup for free data training")
        self.server.model_fit([])
        loss,accuracy=self.server.evaluate()
        print(f"initial server model, accuracy: {accuracy}, loss: {loss}")
        total_syntheticdata=self.server.trainset
        for round in range(rounds):
            self.server.Gan_freedata_fit()
            f=self.server.Gen_fake(round*100)
            total_syntheticdata=torch.cat((total_syntheticdata,f),dim=0).detach()
            print(f"-------------ready for round {round}-------------")
            self.client_M_update()
            round_fake_data=f
            round_fake_data.detach()
            print(f"complete to update client's M model")
            for i in range(self.n_clients):
                fit_data=torch.cat((self.clients[i].trainset,round_fake_data),dim=0)
                print(f"processing client {i}")
                fit_data=fit_data.detach()
                self.clients[i].model_fit(fit_data)
            self.server_M_update()
            loss,accuracy=self.server.evaluate()
            accuracy_hist.append(accuracy)
            print(f"round {round} accuracy for server: {accuracy}")

        loss,accuracy=self.server.evaluate()
        print(f"----------last accuracy {accuracy} ----------")
        return accuracy_hist
    
        

 
                