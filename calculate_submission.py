from torch.utils.data import DataLoader
from dataset import ProjDataset
import torch
import torch.nn as nn
from torchvision import transforms
import os

from mobilenet2 import mobilenet_v2


def calc_submission(checkpoint_dir, embedding_size, similarity, val_transform):

    CHECKPOINT_DIR = os.path.join(checkpoint_dir, "best_checkpoint.pth")

    b_size = 128

    dataset = ProjDataset(which="test", transform=val_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=b_size, shuffle=False, num_workers=8)

    dev = "cuda"

    model = mobilenet_v2(pretrained=False, progress=False, embedding_size=embedding_size)
    checkpoint = torch.load(CHECKPOINT_DIR, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(dev)
    model.eval()
    
    submission_pred = []

    if similarity == "l2":
        d = nn.PairwiseDistance(p=2)
    elif similarity == "cosine":
        d = torch.nn.CosineSimilarity(dim=1, eps=1e-06)
    else:
        raise AssertionError

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            if i % 25 == 0:
                print(f"{i*b_size}\{dataset.__len__()}")
            
            A_imgs = sample[0].to(dev)
            B_imgs = sample[1].to(dev)
            C_imgs = sample[2].to(dev) 

            # Compute the feature embeddings
            A_features = model(A_imgs, test=True)
            B_features = model(B_imgs, test=True)
            C_features = model(C_imgs, test=True)

            preds = 1 * (d(A_features, B_features) < d(A_features, C_features))

            submission_pred.extend(preds.tolist())
    
    with open(os.path.join(checkpoint_dir, "TEST_predictions.txt"), 'w') as fh:
        fh.writelines("%s\n" % str(pred) for pred in submission_pred)