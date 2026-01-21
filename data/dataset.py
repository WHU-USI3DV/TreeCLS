import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import copy
import torch
from torchvision.datasets.folder import default_loader

from .randaug import RandAugment

WHU_CLASS_NAME = ['Acer_palmatum', 'Aesculus_chinensis', 'Ailanthus_altissima', 'Albizia_julibrissin', 'Alstonia_scholaris', 'Araucaria_cunninghamii', 'Archontophoenix_alexandrae', 'Bauhinia_blakeana', 'Bauhinia_purpurea', 'Betula_platyphylla', 'Bischofia_polycarpa', 'Bombax_ceiba', 'Broussonetia_papyrifera', 'Caryota_maxima', 'Catalpa_ovata', 'Cedrus_deodara', 'Ceiba_speciosa', 'Celtis_sinensis', 'Celtis_tetrandra', 'Cinnamomum_camphora', 'Cinnamomum_japonicum', 'Citrus_maxima', 'Cocos_nucifera', 'Delonix_regia', 'Dimocarpus_longan', 'Dracontomelon_duperreanum', 'Elaeocarpus_decipiens', 'Elaeocarpus_glabripetalus', 'Erythrina_variegata', 'Eucommia_ulmoides', 'Euonymus_maackii', 'Euphorbia_milii', 'Ficus_altissima', 'Ficus_benjamina', 'Ficus_concinna', 'Ficus_microcarpa', 'Ficus_virens', 'Firmiana_simplex', 'Fraxinus_chinensis', 'Ginkgo_biloba', 'Grevillea_robusta', 'Koelreuteria_paniculata', 'Lagerstroemia_indica', 'Ligustrum_lucidum', 'Ligustrum_quihoui', 'Liquidambar_formosana', 'Liriodendron_chinense', 'Livistona_chinensis', 'Magnolia_grandiflora', 'Mangifera_persiciforma', 'Melia_azedarach', 'Metasequoia_glyptostroboides', 'Michelia_chapensis', 'Morella_rubra', 'Morus_alba', 'Osmanthus_fragrans', 'Paulownia_tomentosa', 'Phoebe_zhennan', 'Photinia_serratifolia', 'Picea_asperata', 'Picea_koraiensis', 'Picea_meyeri', 'Pinus_elliottii', 'Pinus_tabuliformis', 'Pinus_thunbergii', 'Pittosporum_tobira', 'Platanus', 'Platycladus_orientalis', 'Populus_alba', 'Populus_canadensis', 'Populus_cathayana', 'Populus_davidiana', 'Populus_hopeiensis', 'Populus_nigra', 'Populus_simonii', 'Populus_tomentosa', 'Prunus_cerasifera', 'Prunus_mandshurica', 'Prunus_salicina', 'Pseudolarix_amabilis', 'Pterocarya_stenoptera', 'Quercus_robur', 'Rhododendron_simsii', 'Robinia_pseudoacacia', 'Roystonea_regia', 'Sabina_chinensis', 'Salix_babylonica', 'Salix_matsudana', 'Sapindus_mukorossi', 'Sterculia_lanceolata', 'Styphnolobium_japonicum', 'Syringa_reticulata_subsp_amurensis', 'Syringa_villosa', 'Tilia_amurensis', 'Tilia_mandshurica', 'Toona_sinensis', 'Trachycarpus_fortunei', 'Triadica_sebifera', 'Ulmus_densa', 'Ulmus_pumila', 'Yulania_denudata', 'Zelkova_serrata']

CUB_CLASS_NAME = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']


def build_loader(args):
    train_loader, val_loader, test_loader = None, None, None

    dataset_name = getattr(args, 'dataset', 'WHU')

    if dataset_name == 'WHU':
        train_set = WHUDataset(split='train', args=args, return_index=True)
        val_set = WHUDataset(split='val', args=args, return_index=True)
        test_set = WHUDataset(split='test', args=args, return_index=True)
    elif dataset_name == 'CUB':
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        val_set = ImageDataset(istrain=False, root=args.val_root, data_size=args.data_size, return_index=True)
        test_set = ImageDataset(istrain=False, root=args.val_root, data_size=args.data_size, return_index=True)
    elif dataset_name == 'Aircraft':
        train_set = AircraftDataset(root=args.train_root, train=True, data_size=args.data_size, return_index=True)
        val_set = AircraftDataset(root=args.val_root, train=False, data_size=args.data_size, return_index=True)
        test_set = AircraftDataset(root=args.val_root, train=False, data_size=args.data_size, return_index=True)
    elif dataset_name == 'RSTree':
        train_set = RSTreeDataset(split='train', args=args, return_index=True)
        val_set = RSTreeDataset(split='val', args=args, return_index=True)
        test_set = RSTreeDataset(split='test', args=args, return_index=True)

    train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=args.num_workers, shuffle=False, batch_size=1)

    return train_loader, val_loader, test_loader

def get_dataset(args):
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        return train_set
    return None


class WHUDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, args, return_index: bool = False):
        self.split = split
        self.args = args
        self.data_size = args.data_size
        self.return_index = return_index
        self.feature_type = getattr(args, 'feature_type', None)

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
        )
        ## 使用CLIP的normalize
        # normalize = transforms.Normalize(
        #     mean=[0.48145466, 0.4578275, 0.40821073],
        #     std=[0.26862954, 0.26130258, 0.27577711]
        # )

        if split == 'train':
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((args.data_size, args.data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((args.data_size, args.data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(split)

    def getDataInfo(self, split):
        data_infos = []
        path = f'data/WHU.json'
        f = open(path, 'r')
        data_infos = json.load(f)[split]
        f.close()

        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        image_path = self.args.val_root + '/' + self.data_infos[index]["img_path"]
        label = int(self.data_infos[index]["label"])

        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.

        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)

        if self.return_index:
            return index, img, label

        return img, label


class RSTreeDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, args, return_index: bool = False):
        self.split = split
        self.args = args
        self.data_size = args.data_size
        self.return_index = return_index
        self.feature_type = getattr(args, 'feature_type', None)

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
        )
        ## 使用CLIP的normalize
        # normalize = transforms.Normalize(
        #     mean=[0.48145466, 0.4578275, 0.40821073],
        #     std=[0.26862954, 0.26130258, 0.27577711]
        # )

        if split == 'train':
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((args.data_size, args.data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((args.data_size, args.data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(split)

    def getDataInfo(self, split):
        data_infos = []
        path = f'data/RSTree.json'
        f = open(path, 'r')
        data_infos = json.load(f)[split]
        f.close()

        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        # image_path = self.data_infos[index]["path"]
        image_path = self.args.val_root + '/' + self.data_infos[index]["img_path"]
        label = int(self.data_infos[index]["label"])

        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.

        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)

        if self.return_index:
            return index, img, label

        return img, label


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self,
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
        # normalize = transforms.Normalize(
        #     mean=[0.48145466, 0.4578275, 0.40821073],
        #     std=[0.26862954, 0.26130258, 0.27577711]
        # )

        if istrain:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)

        ## 统计每个类别有多少图片
        class_count = {}
        for info in self.data_infos:
            label = info['label']
            if label not in class_count:
                class_count[label] = 0
            class_count[label] += 1

    def getDataInfo(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        print("[dataset] class number:", len(folders))
        for class_id, folder in enumerate(folders):
            files = os.listdir(root+folder)
            for file in files:
                data_path = root+folder+"/"+file
                data_infos.append({"img_path":data_path, "label":class_id})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["img_path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.

        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)

        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label

        # return img, sub_imgs, label, sub_boundarys
        return img, label


class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, args: object, return_index: bool = False):
        self.split = split
        self.return_index = return_index

        self.args = args
        self.data_size = args.data_size
        self.coarse_map = args.coarse_map
        self.feature_type = getattr(args, 'feature_type', None)

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )

        if split == 'train':
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((self.data_size, self.data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((self.data_size, self.data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(split)

    def getDataInfo(self, split):
        data_dir = os.path.join(self.args.train_root, 'CUB_200_2011')
        image_txt = os.path.join(data_dir, 'images.txt')
        data_infos = []

        with open(image_txt, 'r') as f:
            data = f.readlines()

        if split == 'train':
            flag = '1'
        else:
            flag = '0'

        with open(os.path.join(data_dir, 'train_test_split.txt'), 'r') as f:
            lines = f.readlines()

        for line in lines:
            idx, is_train = line.replace('\n','').split(' ')
            if is_train == flag:
                img_idx, img_path = data[int(idx) - 1].replace('\n','').split(' ')
                assert img_idx == idx
                label = int(img_path.split('.')[0]) - 1
                data_infos.append({"img_path": os.path.join(data_dir, 'images', img_path), "label": label})

        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        # image_path = self.data_infos[index]["path"]
        image_path = self.data_infos[index]["img_path"]
        label = int(self.data_infos[index]["label"])

        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.

        # 5 *768
        ## 为了程序正常进行
        feature = np.ones((5, 768), dtype=np.float32)
        ##
        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)

        if self.return_index:
            if self.coarse_map:
                return index, img, label, label, feature
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label, feature

        # return img, sub_imgs, label, sub_boundarys

        if self.coarse_map:
            return img, label, label, feature
        return img, label, label



class AircraftDataset(torch.utils.data.Dataset):
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, data_size: int, train=True, return_index: bool = False):
        self.train = train
        self.root = root
        self.class_type = 'variant'
        self.split = 'trainval' if self.train else 'test'
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))


        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        if train:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.return_index = return_index

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.transforms(sample)
        if self.return_index:
            return index, sample, target
        return sample, target

    def __len__(self):
        return len(self.samples)

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images