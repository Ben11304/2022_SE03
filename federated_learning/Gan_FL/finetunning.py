import FL
import torch
import pandas as pd 


#select kernel
device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

config = dict(

)

serverdata = pd.read_csv() 
trainloader = pd.read_csv()
testloader = pd.read_csv() 

# simulation_FL = FL.Federated_Learning(config=config, trainloaders=, testloader=, serverdata= , testdata=, device=device)
