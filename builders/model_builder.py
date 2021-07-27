from model.PUNet import PUNet

def build_model(model_name, num_channels):
    if model_name == 'PUNet':
        return PUNet(num_channels=num_channels)