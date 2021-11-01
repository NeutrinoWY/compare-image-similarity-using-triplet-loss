from experiment import Experiment
from torchvision import transforms
import os

from calculate_submission import calc_submission

from PIL import Image

config = {
    "triplet_margin": 0.4,
    "embedding_size": 64,
    "lr" : 0.001,
    "wd" : 0.0001,
    "epochs" : 30,
    "early_stop" : 7,
    "b_size" : 32,
    "n_workers" : 8,
    "use_gpu" : True,
    "debug_mode": False,
    "similarity": "l2"
}


def main():
    tr_transform = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    # transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.1), ratio=(0.95, 1.05), interpolation=2),
                                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
                                    # transforms.RandomHorizontalFlip(p=0.25),
                                    # transforms.RandomRotation(degrees=5, resample=Image.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if not os.path.isdir("experiments"):
        os.mkdir("experiments")

    experiment = Experiment(config=config, train_transform=tr_transform, val_transform=val_transform)

    checkpoint_dir = experiment.train_val_test()

    calc_submission(checkpoint_dir=checkpoint_dir, embedding_size=config["embedding_size"], similarity=config["similarity"], val_transform=val_transform)

if __name__ == "__main__":
    main()