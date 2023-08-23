import torch
import torch.nn as nn
import timm
from train_allsight_regressor.datasets import output_map 
# import resnets as srn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
pc_name = os.getlogin()


def get_model(model_params):

    if model_params['input_type'] == 'single':
        model = PreTrainedModel(model_params['model_name'], output_map[model_params['output']]).to(device)

    elif model_params['input_type'] == 'with_ref_6c':
        model = PreTrainedModelWithRef(model_params['model_name'], output_map[model_params['output']]).to(device)

    # elif model_params['input_type'] == 'with_ref_rnn':
    #     model = srn.resnet18(False, False, num_classes=output_map[model_params['output']]).to(device)

    elif model_params['input_type'] == 'sim_pre_6c':
        model = PreTrainedModelWithRef(model_params['model_name'], output_map['pose']).to(device)
        # path_to_sim_model = '/home/{}/catkin_ws/src/allsight/simulation/train_history/combined/{}'. \
        #     format(pc_name, model_params['model_pre_sim'])
        # model.load_state_dict(torch.load(path_to_sim_model + '/model.pth'))
        model = PreTrainedSimModelWithRef(model, output_map[model_params['output']] - output_map['pose']).to(device)

    elif model_params['input_type'] == 'with_mask_7c':
        model = PreTrainedModelWithMask(model_params['model_name'], output_map[model_params['output']]).to(device)

    elif model_params['input_type'] == 'with_mask_8c':
        model = PreTrainedModelWithOnlineMask(model_params['model_name'], output_map[model_params['output']]).to(device)
    else:
        model = None

    return model

class PreTrainedModel(nn.Module):

    def __init__(self, model_name, num_output, classifier=False, freeze=False):
        super(PreTrainedModel, self).__init__()

        self.is_classifer = classifier
        # self.max_height = 0.026
        self.backbone = self.get_model(freeze=freeze, num_classes=num_output, version=model_name).to(device)

    def get_model(self, version='resnet18', num_classes=3, freeze=False):
        model = timm.create_model(model_name=version, pretrained=True, num_classes=num_classes)

        if freeze:
            for parameter in model.parameters():
                parameter.requires_grad = False

        if version == 'efficientnet_b0':
            in_features = model.classifier.in_features
        else:
            in_features = model.fc.in_features

        # modules = [nn.Linear(in_features, 100),
        #            nn.BatchNorm1d(100),
        #            nn.ReLU(),
        #            nn.Dropout(p=0.3),
        #            nn.Linear(100, num_classes)]

        modules = [nn.Linear(in_features, num_classes)]
        
        if version == 'efficientnet_b0':
            if self.is_classifer:
                model.classifier = nn.Sequential(*modules, nn.Sigmoid())
            else:
                model.classifier = nn.Sequential(*modules)
        else:
            if self.is_classifer:
                model.fc = nn.Sequential(*modules, nn.Sigmoid())
            else:
                model.fc = nn.Sequential(*modules)

        return model

    def forward(self, images, ref_frame=None, masked_img=None, masked_ref=None):

        pred_out = self.backbone(images.to(device))

        return pred_out


class PreTrainedModelWithRef(nn.Module):

    def __init__(self, model_name, num_output, classifer=False, freeze=False):
        super(PreTrainedModelWithRef, self).__init__()

        num_classess = num_output
        self.is_classifer = classifer
        self.backbone = self.get_model(freeze=freeze, num_classes=num_classess, version=model_name).to(device)

        # self.final = nn.Sequential(
        #     nn.Linear(self.in_feature, 100),
        #     nn.BatchNorm1d(100),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(100, num_classess),
        # )
        self.final = nn.Linear(self.in_feature, num_classess)

    def get_model(self, version='resnet18', num_classes=3, freeze=False):

        model = timm.create_model(model_name=version, pretrained=True, num_classes=num_classes)

        if freeze:
            for parameter in model.parameters():
                parameter.requires_grad = False

        if version == 'efficientnet_b0':
            self.in_feature = model.classifier.in_features

            if not self.is_classifer:
                model.classifier = nn.Sequential()
            layer = model.conv_stem
        else:
            self.in_feature = model.fc.in_features

            if not self.is_classifer:
                model.fc = nn.Sequential()            

            layer = model.conv1

        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=6,
                              out_channels=layer.out_channels,
                              kernel_size=layer.kernel_size,
                              stride=layer.stride,
                              padding=layer.padding,
                              bias=layer.bias)

        # Copying the weights from the old to the new layer
        with torch.no_grad():
            new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()
            new_layer.weight[:, layer.in_channels:, :, :] = layer.weight.clone()

            # # Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
            # for i in range(6 - layer.in_channels):
            #     channel = layer.in_channels + i
            #     new_layer.weight[:, channel:channel + 1, :, :] = layer.weight[:, copy_weights:copy_weights + 1, ::].clone()
            new_layer.weight = nn.Parameter(new_layer.weight)
        if version == 'efficientnet_b0':
            model.conv_stem = new_layer
        else:
            model.conv1 = new_layer
        return model

    def forward(self, images, ref_frame=None, masked_img=None, masked_ref=None):

        # x = _diff(images, ref_frame)
        # x = torch.stack((x, x, x), axis=1)
        # diff = images - ref_frame  # torch.unsqueeze(ref_frame, 0).repeat(images.shape[0], 1, 1, 1)
        ref = ref_frame

        x = torch.cat((images, ref), dim=1)
        x1 = self.backbone(x)

        pred_out = self.final(x1)
        return pred_out


class PreTrainedSimModelWithRef(nn.Module):

    def __init__(self, model, num_output, freeze=False):
        super(PreTrainedSimModelWithRef, self).__init__()

        self.backbone = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
        self.pose_layer = model.final
        self.rest_layer = nn.Linear(model.final.in_features, num_output)

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def forward(self, images, ref_frame=None, masked_img=None, masked_ref=None):
        x = torch.cat((images, ref_frame), dim=1)
        features = self.backbone(x)
        out_pose = self.pose_layer(features)
        out_rest = self.rest_layer(features)
        return torch.cat((out_pose, out_rest), dim=1)


class PreTrainedModelWithMask(nn.Module):

    def __init__(self, model_name, num_output, classifer=False, freeze=False):
        super(PreTrainedModelWithMask, self).__init__()

        num_classess = num_output
        self.is_classifer = classifer
        self.backbone = self.get_model(freeze=freeze, num_classes=num_classess, version=model_name).to(device)

        # self.final = nn.Sequential(
        #     nn.Linear(self.in_feature, 100),
        #     nn.BatchNorm1d(100),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(100, num_classess),
        # )
        self.final = nn.Linear(self.in_feature, num_classess)

    def get_model(self, version='resnet18', num_classes=3, freeze=False):

        model = timm.create_model(model_name=version, pretrained=True, num_classes=num_classes)

        if freeze:
            for parameter in model.parameters():
                parameter.requires_grad = False

        self.in_feature = model.fc.in_features

        if not self.is_classifer:
            model.fc = nn.Sequential()

        layer = model.conv1

        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=7,
                              out_channels=layer.out_channels,
                              kernel_size=layer.kernel_size,
                              stride=layer.stride,
                              padding=layer.padding,
                              bias=layer.bias)

        # Copying the weights from the old to the new layer
        with torch.no_grad():
            new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()
            new_layer.weight[:, layer.in_channels:layer.in_channels + 3, :, :] = layer.weight.clone()

            # # Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
            # for i in range(6 - layer.in_channels):
            #     channel = layer.in_channels + i
            #     new_layer.weight[:, channel:channel + 1, :, :] = layer.weight[:, copy_weights:copy_weights + 1, ::].clone()
            new_layer.weight = nn.Parameter(new_layer.weight)

        model.conv1 = new_layer

        return model

    def forward(self, images, ref_frame, masked_img, masked_ref):

        x = torch.cat((images, ref_frame, torch.unsqueeze(masked_ref, 1)), dim=1)
        x1 = self.backbone(x)

        pred_out = self.final(x1)
        return pred_out


class PreTrainedModelWithOnlineMask(nn.Module):

    def __init__(self, model_name, num_output, classifer=False, freeze=False):
        super(PreTrainedModelWithOnlineMask, self).__init__()

        num_classess = num_output
        self.is_classifer = classifer
        self.backbone = self.get_model(freeze=freeze, num_classes=num_classess, version=model_name).to(device)

        # self.final = nn.Sequential(
        #     nn.Linear(self.in_feature, 100),
        #     nn.BatchNorm1d(100),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(100, num_classess),
        # )
        self.final = nn.Linear(self.in_feature, num_classess)

    def get_model(self, version='resnet18', num_classes=3, freeze=False):

        model = timm.create_model(model_name=version, pretrained=True, num_classes=num_classes)

        if freeze:
            for parameter in model.parameters():
                parameter.requires_grad = False

        self.in_feature = model.fc.in_features

        if not self.is_classifer:
            model.fc = nn.Sequential()

        layer = model.conv1

        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=8,
                              out_channels=layer.out_channels,
                              kernel_size=layer.kernel_size,
                              stride=layer.stride,
                              padding=layer.padding,
                              bias=layer.bias)

        # Copying the weights from the old to the new layer
        with torch.no_grad():
            new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()
            new_layer.weight[:, layer.in_channels:layer.in_channels + 3, :, :] = layer.weight.clone()

            # # Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
            # for i in range(6 - layer.in_channels):
            #     channel = layer.in_channels + i
            #     new_layer.weight[:, channel:channel + 1, :, :] = layer.weight[:, copy_weights:copy_weights + 1, ::].clone()
            new_layer.weight = nn.Parameter(new_layer.weight)

        model.conv1 = new_layer

        return model

    def forward(self, images, ref_frame, masked_img, masked_ref):

        # x = _diff(images, ref_frame)
        # x = torch.stack((x, x, x), axis=1)
        # diff = images - ref_frame  # torch.unsqueeze(ref_frame, 0).repeat(images.shape[0], 1, 1, 1)

        x = torch.cat((images, ref_frame, torch.unsqueeze(masked_img, 1), torch.unsqueeze(masked_ref, 1)), dim=1)
        x1 = self.backbone(x)

        pred_out = self.final(x1)
        return pred_out

