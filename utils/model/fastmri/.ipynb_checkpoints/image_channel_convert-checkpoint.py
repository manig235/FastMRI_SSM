import torch

def Channel16To20(input):
    stack_17 = torch.sum(input[:,0:3,...], dim = 1)/4
    stack_18 = torch.sum(input[:,4:7,...], dim = 1)/4
    stack_19 = torch.sum(input[:,8:12,...], dim = 1)/4
    stack_20 = torch.sum(input[:,12:16,...], dim = 1)/4
    return torch.cat([input, stack_17.unsqueeze(1), stack_18.unsqueeze(1), stack_19.unsqueeze(1), stack_20.unsqueeze(1)], axis = 1)

def Channel8To20(input):
    input16 = torch.cat([input, input], axis = 1)
    return Channel16To20(input16)

def Channel14To20(input):
    stack_15 = torch.sum(input[:,0:2,...], dim = 1)/2
    stack_16 = torch.sum(input[:,2:4,...], dim = 1)/2
    stack_17 = torch.sum(input[:,4:6,...], dim = 1)/2
    stack_18 = torch.sum(input[:,6:8,...], dim = 1)/2
    stack_19 = torch.sum(input[:,8:11,...], dim = 1)/3
    stack_20 = torch.sum(input[:,11:14,...], dim = 1)/3
    return torch.cat([input, stack_15.unsqueeze(1), stack_16.unsqueeze(1), stack_17.unsqueeze(1), stack_18.unsqueeze(1), stack_19.unsqueeze(1), stack_20.unsqueeze(1)], axis = 1)

def Channel12To20(input):
    stack_17 = torch.sum(input[:,0:3,...], dim = 1)/3
    stack_18 = torch.sum(input[:,3:6,...], dim = 1)/3
    stack_19 = torch.sum(input[:,6:9,...], dim = 1)/3
    stack_20 = torch.sum(input[:,9:12,...], dim = 1)/3
    return Channel16To20(torch.cat([input, stack_17.unsqueeze(1), stack_18.unsqueeze(1), stack_19.unsqueeze(1), stack_20.unsqueeze(1)], axis = 1))

def Channel4To20(input):
    return Channel16To20(torch.cat([input, input, input, input, input], axis = 1))

def Channel2To20(input):
    return Channel4To20(torch.cat([input, input], axis = 1))

def Channel10To20(input):
    return Channel16To20(torch.cat([input, input], axis = 1))

def Channel18To20(input):
    stack_19 = torch.sum(input[:,:9,...], dim = 1)/9
    stack_20 = torch.sum(input[:,9:,...], dim = 1)/9
    return torch.cat([input, stack_19.unsqueeze(1), stack_20.unsqueeze(1)], axis = 1)

def image_channel_converter(input):
    chans = input.shape[1]
    if chans == 16:
        return Channel16To20(input)
    elif chans == 8:
        return Channel8To20(input)
    elif chans == 14:
        return Channel14To20(input)
    elif chans == 12:
        return Channel12To20(input)
    elif chans == 18:
        return Channel18To20(input)
    elif chans == 4:
        return Channel4To20(input)
    elif chans == 2:
        return Channel2To20(input)
    elif chans == 10:
        return Channel10To20(input)
    return input