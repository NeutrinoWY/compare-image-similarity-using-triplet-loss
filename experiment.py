#from model import MultiModalUnit
from dataset import ProjDataset
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import os
import sys

from mobilenet2 import mobilenet_v2

from util import createDirectory, createCheckpointDir

# Fix random seeds
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)


class Experiment:
    """Comments...."""
    def __init__(self, config, train_transform, val_transform):

        self.train_transform = train_transform
        self.val_transform = val_transform
        
        # Load the configuration
        self.config = config       
        self._check_config_parameters()
        
        # Is the instance for Debugging
        self.debug = config['debug_mode']
        
        # Directory where model checkpoints will be saved
        self.checkpoint_dir = createCheckpointDir(outputFolderPath = "experiments", debug_mode = False)
        
        # Open file for logging
        self.fh = open(os.path.join(self.checkpoint_dir, "log_file.txt"), "a")

        # Create a tensorboard summary writter
        self.summary_writer = self.create_summary_writer()

        # Create the specified model and the optimizer
        self.model, self.dev = self.create_model()  
        self.optimizer, self.lr_scheduler = self.create_optimizer() 

        # Create DataLoaders for the training and validation set
        self.train_loader = self._make_dataloader(which="train") 
        self.val_loader = self._make_dataloader(which="val") 

        # Creates the loss function and the accuracy metric
        self.loss_fn = self.create_loss_fn()

        # Try to load a pre-trained model.
        self.curr_epoch, self.best_epoch, self.val_metric, self.val_loss, self.train_metric, self.train_loss = self.load_helpers()
        
        # Print config file
        self.print_to_log('------------------------------------------')
        for keys,values in self.config.items():
            self.print_to_log( str(keys) + " : " +  str(values) )
        self.print_to_log('------------------------------------------')
        
    def _check_config_parameters(self):
        if not isinstance(self.config["lr"], float):
            raise ValueError
        elif not isinstance(self.config["wd"], float):
            raise ValueError
        elif not isinstance(self.config["b_size"], int):
            raise ValueError
        elif not isinstance(self.config["epochs"], int):
            raise ValueError
        elif not isinstance(self.config["early_stop"], int):
            raise ValueError
        elif not isinstance(self.config["n_workers"], int):
            raise ValueError
        elif not isinstance(self.config["use_gpu"], bool):
            raise ValueError

    def create_model(self):
        if self.config["use_gpu"]: 
            if torch.cuda.is_available():
                dev = "cuda"
            else:
                raise AssertionError
        else:
            dev = "cpu"
        
        model = mobilenet_v2(pretrained=True, progress=False, embedding_size=self.config["embedding_size"])

        return model.to(dev), dev

    def _make_dataloader(self, which):
        if which == "train":
            shuffle = True
            transform = self.train_transform
            b_size = self.config["b_size"]
        elif which == "val":
            shuffle = False
            transform = self.val_transform
            b_size = self.config["b_size"]
        else:
            raise ValueError

        dataset = ProjDataset(which=which, transform=transform)
    
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=b_size, shuffle=shuffle, num_workers=self.config["n_workers"], pin_memory=True)

        return data_loader

    def create_loss_fn(self):
        if self.config["similarity"] == "l2":
            return nn.TripletMarginLoss(margin=self.config["triplet_margin"], p=2)
        elif self.config["similarity"] == "cosine":
            return TripletCosineLoss(margin=self.config["triplet_margin"])
        else:
            raise AssertionError

    def create_optimizer(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["wd"], momentum=0.9)
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)

        return optimizer, lr_scheduler

    def create_summary_writer(self):
        return SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, "tensorboard_logs"))

    def train_val_test(self):
        # To meassure average epoch processing time
        epoch_time = AverageMeter()
        # Time of the beginning of an epoch
        start_time = time.time()  

        for curr_epoch in range(1, self.config["epochs"] + 1): 
            # Update current epoch count
            self.curr_epoch = curr_epoch

            # Log the estimated time for finishing the experiment
            self._log_estimated_epoch_time(epoch_time=epoch_time, lr=self.optimizer.param_groups[0]["lr"])

            # Train for one epoch
            train_loss, train_metric = self.train_epoch()

            # Validate model on the validation set
            val_loss, val_metric = self.validate_epoch()

            # Adjust the learning rate eafter each epoch, according to the lr scheduler
            self.lr_scheduler.step(val_loss)

            if self.summary_writer is not None:
                # Log metrics to the tensorboard
                self.summary_writer.add_scalars(f"Loss",
                                                {"train": train_loss, "val": val_loss},
                                                self.curr_epoch)
                self.summary_writer.add_scalars(f"Metric",
                                                {"train": train_metric, "val": val_metric},
                                                self.curr_epoch)

            # Calculate epoch time, and restart timer
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

            # Save a checkpoint of the model after each epoch
            self.save_checkpoint()

            # Check if the training should stop due to early stopping
            if (self.curr_epoch - self.best_epoch) == self.config["early_stop"]: 
                self.print_to_log("EARLY STOPPING \n")
                break

        # Log the training report 
        self.training_end_report()

        # Close the file for logging
        self.fh.close()

        return self.checkpoint_dir 
    
    def train_epoch(self):
        # Train for one epoch
        train_loss, train_metric = self.run_one_epoch(which="train", update_weights=True)

        # Append the loss and acc to the list (save one value for evey epoch)
        self.train_loss.append(train_loss)
        self.train_metric.append(train_metric)

        return train_loss, train_metric

    def validate_epoch(self):
        # Validate for one epoch
        val_loss, val_metric = self.run_one_epoch(which="val", update_weights=False)

        # If the current validation loss is better than from all previous epochs
        if self.curr_epoch == 1:
            self.best_epoch = 1
        elif val_loss < np.min(self.val_loss):
            self.best_epoch = self.curr_epoch

        # Append the loss and acc to the list (save one value for evey epoch)
        self.val_loss.append(val_loss)
        self.val_metric.append(val_metric)

        return val_loss, val_metric

    def run_one_epoch(self, which, update_weights):

        assert isinstance(update_weights, bool)

        # Take the specified data loader
        if which == "train":
            data_loader = self.train_loader
            split_name = "Training"
        elif which == "val":
            data_loader = self.val_loader
            split_name = "Validation"
        else:
            raise AssertionError

        # Put the model in the appropriate mode. 
        # If it should update weights, put in training mode. If not, in evaluation mode.
        if update_weights:
            self.model.train()  
        else:
            self.model.eval() 

        # For averaging batch processing times over the epoch
        batch_time = AverageMeter() 
        # For averaging data loading time over the epoch
        data_time = AverageMeter() 
        # For averaging losses over the epoch
        losses = AverageMeter()
        # For storing all (prediction, target) pairs in the epoch 
        metrics = AverageMeter()

        # Measure the beginning of the batch (and also beginiing of data loading)
        batch_start_time = time.time()  
        # Loop over the whole dataset once
        for i, sample in enumerate(data_loader):

            if i % 50 == 0:
                self.print_to_log(f"{i*sample[0].shape[0]}\ approx. {int(59515*0.9 if which=='train' else 59515*0.1)}")

            if which == "train":
                if i*self.config["b_size"] >= 10000:
                    break

            # TODO Debugging better
            
            A_imgs = sample[0].to(self.dev)
            B_imgs = sample[1].to(self.dev)
            C_imgs = sample[2].to(self.dev)

            # Measure data loading/pre-processing time
            data_time.update(time.time() - batch_start_time)  

            # Compute the feature embeddings
            A_features = self.model(A_imgs)
            B_features = self.model(B_imgs)
            C_features = self.model(C_imgs) 

            # Compute the loss 
            loss = self.loss_fn(anchor=A_features, positive=B_features, negative=C_features) 

            # Delete calculated gradients
            self.optimizer.zero_grad()  

            # If in weight update mode
            if update_weights:
                # Calculate the loss gradients
                loss.backward()
                # Update network weights with calculated gradients  
                self.optimizer.step()
                # Delete calculated gradients
                self.optimizer.zero_grad()   

            # Update epoch loss averaging
            losses.update(loss.item())  

            # Update accuracy metric calculator
            with torch.no_grad():
                metric = self.calculate_metric(a=A_features, p=B_features, n=C_features)

                # Update epoch metric averaging
                metrics.update(metric.item())
            
            del sample, A_imgs, A_features, B_imgs, B_features, C_imgs, C_features, loss, metric
            torch.cuda.empty_cache()

            # Measure the time it took to process the batch
            batch_time.update(time.time() - batch_start_time)
            # Measure the beginning of the next ba
            batch_start_time = time.time()  

        # Calculate average epoch loss
        loss_avg = losses.get_average()
        # Calculate the average of the metric on this epoch
        metric_avg = metrics.get_average()

        self.print_to_log(f"{split_name} loss: {loss_avg:.6f}")
        self.print_to_log(f"{split_name} metric: {metric_avg}")
        
        return loss_avg, metric_avg

    def calculate_metric(self, a, p, n):
        if self.config["similarity"] == "l2":
            d = nn.PairwiseDistance(p=2)
        elif self.config["similarity"] == "cosine":
            d = torch.nn.CosineSimilarity(dim=1, eps=1e-06)
        else:
            raise AssertionError
        metric = torch.mean(1.0 * (d(a, p) < d(a, n)))
        return metric

    def load_helpers(self):
        curr_epoch = 1
        best_epoch = 1
        val_metric = []
        val_loss = []
        train_metric = []
        train_loss = []

        return curr_epoch, best_epoch, val_metric, val_loss, train_metric, train_loss

    def save_checkpoint(self):
        self.model.eval()  # Switch the model to evaluation mode
        state = {'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'curr_epoch': self.curr_epoch,
                 'best_epoch': self.best_epoch,
                 'train_metric': self.train_metric,
                 'train_loss': self.train_loss,
                 'val_metric': self.val_metric,
                 'val_loss': self.val_loss}
        torch.save(state, os.path.join(self.checkpoint_dir, "checkpoint.pth"))
        if self.curr_epoch == self.best_epoch:
            torch.save(state, os.path.join(self.checkpoint_dir, "best_checkpoint.pth"))

        return
    
    def print_to_log(self, message):
        print(message, file=self.fh)
        print(message)
        self.fh.flush()

    def training_end_report(self):
        # Best epoch indexes to extract best metrics from arrays
        best_tr_epoch_i = int(np.argmin(self.train_loss))
        best_val_epoc_i = self.best_epoch - 1  # Minus one because epochs start from 1, and list indexing starts from 0

        self.print_to_log(" ")
        self.print_to_log("---------------------------------------------------------------------------------")
        self.print_to_log("End of training report:")
        self.print_to_log("---------------------------------------------------------------------------------")
        self.print_to_log("Best training loss: {:0.4f}".format(self.train_loss[best_tr_epoch_i]))
        self.print_to_log("Best training metric: {:0.4f}".format(self.train_metric[best_tr_epoch_i]))
        self.print_to_log("Best validation loss: {:0.4f}".format(self.val_loss[best_val_epoc_i]))
        self.print_to_log("Best validation metric: {:0.4f}".format(self.val_metric[best_val_epoc_i]))
        self.print_to_log("Epoch with the best training loss: {}".format(best_tr_epoch_i + 1))
        self.print_to_log("Epoch with the best validation loss: {}".format(self.best_epoch))
        self.print_to_log("---------------------------------------------------------------------------------")
        self.print_to_log(" ")
        self.print_to_log("Finished training. Starting the evaluation.")
        self.print_to_log("---------------------------------------------------------------------------------")
        self.print_to_log(" ")
        return
    
    def _log_estimated_epoch_time(self, epoch_time, lr):
        """
        self.Print_to_logself.print_to_log the estimated time to finish this experiment, as well as the lr for the current epoch.

        :param epoch_time:      average time per one epoch
        :param lr:              current lr
        """
        # Info about the last epoch
        # (Do not self.print_to_log before first epoch happens)
        if epoch_time.val != 0.0:

            epoch_h, epoch_m, epoch_s = convert_secs2time(epoch_time.val)
            self.print_to_log('Epoch processing time: {:02d}:{:02d}:{:02d} (H:M:S) \n'.format(epoch_h, epoch_m, epoch_s))

        # Info about the beginning of the current epoch
        remaining_seconds = epoch_time.get_average() * (self.config["epochs"] - self.curr_epoch)  
        need_hour, need_mins, _ = convert_secs2time(remaining_seconds)
        need_time = '[Need: {:02d}:{:02d} (H:M)]'.format(need_hour, need_mins)
        self.print_to_log('{:3d}/{:3d} ----- [{:s}] {:s} LR={:}'.format(self.curr_epoch, self.config["epochs"], time_string(), need_time, lr))


class TripletCosineLoss(nn.Module):
    "Taken from https://discuss.pytorch.org/t/triplet-loss-in-pytorch/30634"
    def __init__(self, margin=1.0):
        super(TripletCosineLoss, self).__init__()

        self.margin = margin
    
    def forward(self, anchor, positive, negative) : 
        d = nn.CosineSimilarity(dim=1, eps=1e-6)
        distance = d(anchor, positive) - d(anchor, negative) + self.margin 
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def get_average(self):
        return self.avg

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs

def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string
