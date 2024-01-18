from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import gym
from torchvision import transforms
from .robotic_transformer_pytorch import MaxViT, RT1
from .transformer import RT1_Mod


# class RT1Extractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict, device):
#         super().__init__(observation_space, features_dim=1)
#         extractors = {}
#         total_concat_size = 0
#         feature_size = 256
#         device = torch.device("cuda", device)

#         for key, subspace in observation_space.spaces.items():
#             if key == "observation":
#                 # vit = MaxViT(
#                 #     num_classes = 1000,
#                 #     dim_conv_stem = 64,
#                 #     dim = 96,
#                 #     dim_head = 32,
#                 #     depth = (2, 2, 5, 2),
#                 #     window_size = 7,
#                 #     mbconv_expansion_rate = 4,
#                 #     mbconv_shrinkage_rate = 0.25,
#                 #     dropout = 0.1
#                 # )

#                 # RT1model = RT1(
#                 #     vit = vit,
#                 #     depth = 6,
#                 #     heads = 8,
#                 #     dim_head = 64,
#                 #     cond_drop_prob = 0.2,
#                 #     conditioner_kwargs = dict(
#                 #         model_types = 't5',
#                 #         model_names = r'/home/amax/Project/Pretrained/t5-v1_1-base',
#                 #     )
#                 # )
#                 RT1model = RT1_Mod()
#                 fc = nn.Sequential(nn.Linear(768, feature_size), nn.ReLU())
#                 self.RT1=RT1model.to(device)
#                 extractors["observation"] = fc

#                 total_concat_size += feature_size
#             if key == "state":
#                 state_size = subspace.shape[0]
#                 extractors["state"] = nn.Linear(state_size, 64)
#                 total_concat_size += 64

#         self.extractors = nn.ModuleDict(extractors).to(device)
#         print('model device: ', next(RT1model.parameters()).device)
        
#         # Update the features dim manually
#         self._features_dim = total_concat_size

#         self.transform=transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )

    # def forward(self, observations) -> torch.Tensor:
    #     # print(observations)

    #     encoded_tensor_list = []

    #     for key, extractor in self.extractors.items():
    #         if key == "observation":
    #             image = observations[key].permute((0, 3, 1, 2))
    #             image = self.transform(image)
    #             image = image.unsqueeze(2)

    #             # instruction=['pick']

    #             # logits=self.RT1(image,instruction)

    #             logits=self.RT1(image)

    #             features = extractor(logits)
    #             if torch.isnan(features).any():
    #                 print('features of rgb appear nan')
    #                 print('image',image)
    #             encoded_tensor_list.append(features)
    #         elif key == "state":
    #             inputs = observations[key]
    #             features=extractor(inputs)
    #             encoded_tensor_list.append(features)
    #             if torch.isnan(features).any():
    #                 print('features of state appear nan')
    #                 print('state',observations[key])

    #     return torch.cat(encoded_tensor_list, dim=1)

class RT1Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, device):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        feature_size = 256
        device = torch.device("cuda", device)

        RT1model = RT1_Mod()

        self.RT1=RT1model.to(device)
        total_concat_size += feature_size
        self.extractor = nn.Sequential(nn.Linear(768, feature_size), nn.ReLU()).to(device)
        print('model device: ', next(RT1model.parameters()).device)
        
        # Update the features dim manually
        self._features_dim = total_concat_size

        self.transform=transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, observations) -> torch.Tensor:
        # print(observations)

        image = observations['observation'].permute((0, 3, 1, 2))
        image = self.transform(image)
        image = image.unsqueeze(2)

        state = observations['state']

        logits=self.RT1(image, state)
        features = self.extractor(logits)
        
        if torch.isnan(features).any():
            print('features of rgb appear nan')

        return features