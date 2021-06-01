import logging
import os
import numpy as np
from numpy.lib.type_check import imag
import torch
import torch.optim as optim
import tensorboard_logger
from tensorboard_logger import log_value

from utils.read_config import Config
from utils.loss import total_loss, chamfer
from dataset import CAD_train_dataloader, CAD_valid_dataloader, CAD_test_dataloader
from models.UCSGNet import UCSGNet
from tqdm import tqdm

#Config parameters, set logging
config = Config("cad_config.yml")

model_name = config.model_path.format(config.lr, config.latent_sz, config.num_csg_layer, config.num_dim)
config.model_path = model_name
print(config.config, flush=True)

logger = logging.getLogger(__name__)
if config.debug == False:
    config.write_config("log/configs/{}_config.txt".format(model_name))
    tensorboard_logger.configure("log/tensorboard/{}".format(model_name), flush_secs=5)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    file_handler = logging.FileHandler('log/logger/{}.log'.format(model_name), mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(config.config)

train_dataset = CAD_train_dataloader(config)
valid_dataset = CAD_valid_dataloader(config)
#test_dataset = CAD_test_dataloader(config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU')

net = UCSGNet(config)
net.to(device)

if config.preload_model:
    print(config.pretrain_modelpath, "Loaded")
    net.load_state_dict(torch.load(config.pretrain_modelpath))

if config.optim == "adam":
    optimizer = optim.Adam(net.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
elif config.optim == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=config.lr)

# Create the output directory.
if not os.path.exists('trained_models'):
    os.makedirs('trained_models')

#Training, Testing of unsupervised learning
prev_CD = 16

for epoch in range(config.epochs):
    
    net.train()
    pbar = tqdm(total=len(train_dataset.dataset), leave=False)
    epoch_str = '' if epoch is None else '[Epoch {}/{}]'.format(
            str(epoch).zfill(len(str(config.epochs))), config.epochs)

    train_loss = 0.0
    n = 0.0

    for batch in train_dataset:
        optimizer.zero_grad()
        
        image, pt, distances, bounding_volume = batch
        image = image.to(device)
        pt = pt.to(device)
        distances = distances.to(device)
        
        pred = net(image, pt)
        loss, each_loss = total_loss(pred, distances, net.converter, net.csg_layer, 
                                            net.evaluator, config.use_planes) 
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        n += 1
 
        pbar.set_description('{} {} Loss: {:f}'.format(epoch_str, 'Train', loss.item()))
        pbar.update(image.shape[0])

    pbar.close()
    mean_train_loss = train_loss / n

    print("Epoch {}/{} => train_loss: {}".format(epoch, config.epochs, mean_train_loss))
    log_value('train_loss', mean_train_loss, epoch)
    
    net.eval()
    pbar = tqdm(total=len(valid_dataset.dataset), leave=False)

    valid_loss = 0.0
    valid_CD = 0.0
    n = 0.0
    for batch in valid_dataset:
        with torch.no_grad():
            image, pt, distances, bounding_volume = batch
            image = image.to(device)
            pt = pt.to(device)
            distances = distances.to(device)
            pred = net(image, pt)

            loss, each_loss = total_loss(pred, distances, net.converter, net.csg_layer, 
                                            net.evaluator, config.use_planes) 
            valid_loss += loss.item()
            n += 1
            CD = chamfer(net.binarize(pred).reshape(-1,64,64).clone().cpu().numpy(),
                                image.squeeze().clone().cpu().numpy()) / image.shape[0]
            valid_CD += CD
        pbar.set_description('{} {} Loss: {:f}, CD: {:f}'.format(epoch_str, 'Valid', loss.item(), CD))
        pbar.update(image.shape[0])

    pbar.close()

    mean_valid_loss = valid_loss / n
    valid_CD = valid_CD / n 
    log_value('valid_loss', mean_valid_loss, epoch)
    log_value('chamfer_distance', valid_CD, epoch)

    logger.info("Epoch {}/{} => valid_loss: {:f}, CD: {:f}".format(epoch, config.epochs, mean_valid_loss, valid_CD))
    print("Epoch {}/{} => valid_loss: {:f}, CD: {:f}".format(epoch, config.epochs, mean_valid_loss, valid_CD))

    if prev_CD > valid_CD:
        logger.info("Saving the Model based on Chamfer Distance: %f"%(valid_CD))
        print("Saving the Model based on Chamfer Distance: %f"%(valid_CD), flush=True)
        torch.save(net.state_dict(), "trained_models/{}.pth".format(model_name))
        prev_CD = valid_CD