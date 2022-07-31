import torch
import torch.nn as nn

# Syntax
# can use tensor model or dictionary parameter
# torch.save(arg, PATH)
# torch.load(PATH)
# model.load_state_dict(arg)
# model.eval() # to evaluate the model

# save parameter
#torch.save(model.state_dict(), PATH)

# model must be created again with parameters
# model = Model(*arg,**kwargs)
# model.load_state_dict(torch.load(path))
# model.eval()

class Model(nn.Module):
    def __init__(self, n_input_feature):
        super(Model,self).__init__()
        self.linear = nn.Linear(n_input_feature, 1)

    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = Model(n_input_feature = 6)

for param in model.parameters():
        print(param)

# pth for pytorch file
file_name = "model.pth"
# torch.save(model,file_name)
# save model parameter
# torch.save(model.state_dict(),file_name)

# load model:
# model = torch.load(file_name)
# model.eval()

# for param in model.parameters():
#         print(param)

# # for load state_dict
# loaded_model = Model(n_input_feature=6)
# loaded_model.load_state_dict(torch.load(file_name))
# model.eval()
#
# for param in loaded_model.parameters():
#         print(param)

# load model again
# model = Model(n_input_feature = 6)
# print(model.state_dict())

# saving whole check point in the
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# output a dictionary
print(optimizer.state_dict())

# if we wait to stop the training and save the checkpoint
# mustbe dictionary

checkpoint = {
    "eport": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}
# torch.save(checkpoint,"checkpoint.pth")

# Load checkpoint
loaded_checkpoint = torch.load("checkpoint.pth")
eport = loaded_checkpoint["eport"]

model = Model(n_input_feature = 6)
optimizer = torch.optim.SGD(model.parameters(), lr=0) # will load the correct lr from the model saved

model.load_state_dict(checkpoint["model_state"])
print(eport)
optimizer.load_state_dict(checkpoint["optim_state"])
print(optimizer.state_dict())

# Save model of GPU, Load on CPU
# device = torch.device("cuda")
# model.to(device)
# torch.save(model.state_dict(), file_name)
# device = torch.device('cpu')
# # create the model again (parmaeter)
# model = Model(*args, **kwargs)
# model.load_state_dict(torch.load(file_name, map_location=device))

# Save on GPU and Load on GPU
# device = torch.device("cuda")
# model.to(device)
# torch.save(model.state_dict(), file_name)
#
# model = Model(*args, **kwargs)
# model.load_state_dict(torch.load(file_name))
# model.to(device)