import torch
from typing import Union
from torchvision.models.feature_extraction import get_graph_node_names

from .pim_module import pim_module

"""
[Default Return]
Set return_nodes to None, you can use default return type, all of the model in this script
return four layers features.

[Model Configuration]
if you are not using FPN module but using Selector and Combiner, you need to give Combiner a
projection  dimension ('proj_size' of GCNCombiner in pim_module.py), because graph convolution
layer need the input features dimension be the same.

[Combiner]
You must use selector so you can use combiner.

[About Costom Model]
This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy
could cause error, so we set return_nodes to None and change swin-transformer model script to
return features directly.
Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
model also fail at create_feature_extractor or get_graph_node_names step.
"""

def load_model_weights(model, model_path):
    ### reference https://github.com/TACJu/TransFG
    ### thanks a lot.
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model


def build_resnet50(pretrained: str = "./resnet50_miil_21k.pth",
                   num_selects: Union[dict, None] = None,
                   img_size: int = 384,
                   use_fpn: bool = True,
                   fpn_size: int = 512,
                   proj_type: str = "Linear",
                   upsample_type: str = "Conv",
                   use_selection: bool = True,
                   num_classes: int = 200,
                   use_combiner: bool = True,
                   comb_proj_size: Union[int, None] = None,
                   coarse_classes: Union[int, None] = None,
                   add_linear: bool = False,
                   feature_type: Union[str, None] = None,
                   add_loss: bool = False,
                   only_loss: bool = False,
                   no_mask: bool = False,
                   use_embedding: bool = False,
                   pretrained_path: Union[str, None] = None,
                   use_cam: bool = False,
                   num_bins: int = 8,
                   fuse_type: int = 0):

    import timm

    return_nodes = {
        'layer1.2.act3': 'layer1',
        'layer2.3.act3': 'layer2',
        'layer3.5.act3': 'layer3',
        'layer4.2.act3': 'layer4',
    }
    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }

    backbone = timm.create_model('resnet50', pretrained=True, num_classes=11221)

    # print(backbone)
    # print(get_graph_node_names(backbone))

    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                    img_size = img_size,
                                    use_fpn = use_fpn,
                                    fpn_size = fpn_size,
                                    proj_type = proj_type,
                                    upsample_type = upsample_type,
                                    use_selection = use_selection,
                                    num_classes = num_classes,
                                    num_selects = num_selects,
                                    use_combiner = num_selects,
                                    comb_proj_size = comb_proj_size,
                                    coarse_classes = coarse_classes,
                                    add_linear = add_linear,
                                    feature_type = feature_type,
                                    add_loss= add_loss,
                                    only_loss= only_loss,
                                    no_mask= no_mask,
                                    use_embedding= use_embedding,
                                    pretrained_path= pretrained_path,
                                    use_cam= use_cam,
                                    num_bins= num_bins,
                                    fuse_type= fuse_type)


def build_vit16(pretrained: str = "./vit_base_patch16_224_miil_21k.pth",
                num_selects: Union[dict, None] = None,
                img_size: int = 384,
                use_fpn: bool = True,
                fpn_size: int = 512,
                proj_type: str = "Linear",
                upsample_type: str = "Conv",
                use_selection: bool = True,
                num_classes: int = 200,
                use_combiner: bool = True,
                comb_proj_size: Union[int, None] = None,
                coarse_classes: Union[int, None] = None,
                add_linear: bool = False,
                feature_type: Union[str, None] = None,
                add_loss: bool = False,
                only_loss: bool = False,
                no_mask: bool = False,
                use_embedding: bool = False,
                pretrained_path: Union[str, None] = None,
                use_cam: bool = False,
                num_bins: int = 8,
                fuse_type: int = 0):

    import timm

    backbone = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=True)
    ### original pretrained path "./models/vit_base_patch16_224_miil_21k.pth"

    backbone.train()

    # print(backbone)
    # print(get_graph_node_names(backbone))
    # 0~11 under blocks

    return_nodes = {
        'blocks.8': 'layer1',
        'blocks.9': 'layer2',
        'blocks.10': 'layer3',
        'blocks.11': 'layer4',
    }
    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }

    ### Vit model input can transform 224 to another, we use linear
    ### thanks: https://github.com/TACJu/TransFG/blob/master/models/modeling.py
    import math
    from scipy import ndimage

    posemb_tok, posemb_grid = backbone.pos_embed[:, :1], backbone.pos_embed[0, 1:]
    posemb_grid = posemb_grid.detach().numpy()
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = img_size//16
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    backbone.pos_embed = torch.nn.Parameter(posemb)

    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                    img_size = img_size,
                                    use_fpn = use_fpn,
                                    fpn_size = fpn_size,
                                    proj_type = proj_type,
                                    upsample_type = upsample_type,
                                    use_selection = use_selection,
                                    num_classes = num_classes,
                                    num_selects = num_selects,
                                    use_combiner = num_selects,
                                    comb_proj_size = comb_proj_size,
                                    coarse_classes = coarse_classes,
                                    add_linear = add_linear,
                                    feature_type = feature_type,
                                    add_loss= add_loss,
                                    only_loss= only_loss,
                                    no_mask= no_mask,
                                    use_embedding= use_embedding,
                                    pretrained_path= pretrained_path,
                                    use_cam= use_cam,
                                    num_bins= num_bins,
                                    fuse_type= fuse_type,
                                    use_cls_token=True)


def build_bioclip2(pretrained: str = "./vit_base_patch16_224_miil_21k.pth",
                num_selects: Union[dict, None] = None,
                img_size: int = 384,
                use_fpn: bool = True,
                fpn_size: int = 512,
                proj_type: str = "Linear",
                upsample_type: str = "Conv",
                use_selection: bool = True,
                num_classes: int = 200,
                use_combiner: bool = True,
                comb_proj_size: Union[int, None] = None,
                coarse_classes: Union[int, None] = None,
                add_linear: bool = False,
                feature_type: Union[str, None] = None,
                add_loss: bool = False,
                only_loss: bool = False,
                no_mask: bool = False,
                use_embedding: bool = False,
                pretrained_path: Union[str, None] = None,
                use_cam: bool = False,
                num_bins: int = 8,
                fuse_type: int = 0):

    from models import open_clip
    model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    backbone = model.visual

    backbone.train()

    return_nodes = {
        'transformer.resblocks.5': 'layer1',
        'transformer.resblocks.11': 'layer2',
        'transformer.resblocks.17': 'layer3',
        'transformer.resblocks.23': 'layer4',
    }
    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }

    ### Vit model input can transform 224 to another, we use linear
    ### thanks: https://github.com/TACJu/TransFG/blob/master/models/modeling.py
    import math
    from scipy import ndimage

    positional_embedding = backbone.positional_embedding.unsqueeze(0)
    posemb_tok, posemb_grid = positional_embedding[:, :1], positional_embedding[0, 1:]
    posemb_grid = posemb_grid.detach().numpy()
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = img_size//14
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1).squeeze(0)
    backbone.positional_embedding = torch.nn.Parameter(posemb)

    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = return_nodes,
                                    img_size = img_size,
                                    use_fpn = use_fpn,
                                    fpn_size = fpn_size,
                                    proj_type = proj_type,
                                    upsample_type = upsample_type,
                                    use_selection = use_selection,
                                    num_classes = num_classes,
                                    num_selects = num_selects,
                                    use_combiner = num_selects,
                                    comb_proj_size = comb_proj_size,
                                    coarse_classes = coarse_classes,
                                    add_linear = add_linear,
                                    feature_type = feature_type,
                                    add_loss= add_loss,
                                    only_loss= only_loss,
                                    no_mask= no_mask,
                                    use_embedding= use_embedding,
                                    pretrained_path= pretrained_path,
                                    use_cam= use_cam,
                                    num_bins= num_bins,
                                    fuse_type= fuse_type,
                                    use_cls_token=True)




def build_swintransformer(pretrained: bool = True,
                          num_selects: Union[dict, None] = None,
                          img_size: int = 384,
                          use_fpn: bool = True,
                          fpn_size: int = 512,
                          proj_type: str = "Linear",
                          upsample_type: str = "Conv",
                          use_selection: bool = True,
                          num_classes: int = 200,
                          use_combiner: bool = True,
                          comb_proj_size: Union[int, None] = None,
                          coarse_classes: Union[int, None] = None,
                          add_linear: bool = False,
                          feature_type: Union[str, None] = None,
                          add_loss: bool = False,
                          only_loss: bool = False,
                          no_mask: bool = False,
                          use_embedding: bool = False,
                          pretrained_path: Union[str, None] = None,
                          use_cam: bool = False,
                          num_bins: int = 8,
                          fuse_type: int = 0):
    """
    This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy
    could cause error, so we set return_nodes to None and change swin-transformer model script to
    return features directly.
    Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
    model also fail at create_feature_extractor or get_graph_node_names step.
    """

    import timm

    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }
    # swin_base_patch4_window12_384_in22k
    backbone = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=True)

    # print(backbone)
    # print(get_graph_node_names(backbone))
    backbone.train()

    print("Building...")
    return pim_module.PluginMoodel(backbone = backbone,
                                   return_nodes = None,
                                    img_size = img_size,
                                    use_fpn = use_fpn,
                                    fpn_size = fpn_size,
                                    proj_type = proj_type,
                                    upsample_type = upsample_type,
                                    use_selection = use_selection,
                                    num_classes = num_classes,
                                    num_selects = num_selects,
                                    use_combiner = num_selects,
                                    comb_proj_size = comb_proj_size,
                                    coarse_classes = coarse_classes,
                                    add_linear = add_linear,
                                    feature_type = feature_type,
                                    add_loss= add_loss,
                                    only_loss= only_loss,
                                    no_mask= no_mask,
                                    use_embedding= use_embedding,
                                    pretrained_path= pretrained_path,
                                    use_cam= use_cam,
                                    num_bins= num_bins,
                                    fuse_type= fuse_type)



def build_lcmodel(pretrained: bool = True,
                num_selects: Union[dict, None] = None,
                img_size: int = 384,
                use_fpn: bool = True,
                fpn_size: int = 512,
                proj_type: str = "Linear",
                upsample_type: str = "Conv",
                use_selection: bool = True,
                num_classes: int = 200,
                use_combiner: bool = True,
                comb_proj_size: Union[int, None] = None,
                coarse_classes: Union[int, None] = None,
                add_linear: bool = False,
                feature_type: Union[str, None] = None,
                add_loss: bool = False,
                only_loss: bool = False,
                no_mask: bool = False,
                use_embedding: bool = False,
                pretrained_path: Union[str, None] = None,
                use_cam: bool = False,
                num_bins: int = 8,
                fuse_type: int = 0):
    """
    This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy
    could cause error, so we set return_nodes to None and change swin-transformer model script to
    return features directly.
    Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
    model also fail at create_feature_extractor or get_graph_node_names step.
    """

    import timm

    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }

    # swin_base_patch4_window12_384_in22k
    if num_classes == 200:
        backbone = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True)
    else:
        backbone = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=True)

    # print(backbone)
    # print(get_graph_node_names(backbone))
    backbone.train()

    print("Building...")
    from .lc_module.model import LCModel
    return LCModel(backbone = backbone,
                    return_nodes = None,
                    img_size = img_size,
                    use_fpn = use_fpn,
                    fpn_size = fpn_size,
                    proj_type = proj_type,
                    upsample_type = upsample_type,
                    use_selection = use_selection,
                    num_classes = num_classes,
                    num_selects = num_selects,
                    use_combiner = num_selects,
                    comb_proj_size = comb_proj_size,
                    coarse_classes = coarse_classes,
                    add_linear = add_linear,
                    feature_type = feature_type,
                    add_loss= add_loss,
                    only_loss= only_loss,
                    no_mask= no_mask,
                    use_embedding= use_embedding,
                    pretrained_path= pretrained_path,
                    use_cam= use_cam,
                    num_bins= num_bins,
                    fuse_type= fuse_type
                    )


def build_herbs_model(pretrained: bool = True,
                num_selects: Union[dict, None] = None,
                img_size: int = 384,
                use_fpn: bool = True,
                fpn_size: int = 512,
                proj_type: str = "Linear",
                upsample_type: str = "Conv",
                use_selection: bool = True,
                num_classes: int = 200,
                use_combiner: bool = True,
                comb_proj_size: Union[int, None] = None,
                coarse_classes: Union[int, None] = None,
                add_linear: bool = False,
                feature_type: Union[str, None] = None,
                add_loss: bool = False,
                only_loss: bool = False,
                no_mask: bool = False,
                use_embedding: bool = False,
                pretrained_path: Union[str, None] = None,
                use_cam: bool = False,
                num_bins: int = 8,
                fuse_type: int = 0):
    """
    This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy
    could cause error, so we set return_nodes to None and change swin-transformer model script to
    return features directly.
    Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
    model also fail at create_feature_extractor or get_graph_node_names step.
    """

    import timm

    if num_selects is None:
        num_selects = {
            'layer1':32,
            'layer2':32,
            'layer3':32,
            'layer4':32
        }

    # swin_base_patch4_window12_384_in22k
    if num_classes == 200:
        backbone = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True)
    else:
        backbone = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=True)

    # print(backbone)
    # print(get_graph_node_names(backbone))
    backbone.train()

    print("Building...")
    from .herbs_module.model import LCModel
    return LCModel(backbone = backbone,
                    return_nodes = None,
                    img_size = img_size,
                    use_fpn = use_fpn,
                    fpn_size = fpn_size,
                    proj_type = proj_type,
                    upsample_type = upsample_type,
                    use_selection = use_selection,
                    num_classes = num_classes,
                    num_selects = num_selects,
                    use_combiner = num_selects,
                    comb_proj_size = comb_proj_size,
                    coarse_classes = coarse_classes,
                    add_linear = add_linear,
                    feature_type = feature_type,
                    add_loss= add_loss,
                    only_loss= only_loss,
                    no_mask= no_mask,
                    use_embedding= use_embedding,
                    pretrained_path= pretrained_path,
                    use_cam= use_cam,
                    )


MODEL_GETTER = {
    "resnet50":build_resnet50,
    "swin-t":build_swintransformer,
    "vit":build_vit16,
    "bioclip":build_bioclip2,
    "lc-model":build_lcmodel,
    "herbs-model":build_herbs_model
}
