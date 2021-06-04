import logging
import os
import torch
import torch.nn.functional
import torch.optim as optim
import tensorboard_logger
from tensorboard_logger import log_value

from utils.read_config import Config
from utils.loss import total_loss
from dataset import ShapeNet_train_dataloader, ShapeNet_valid_dataloader
from models.UCSGNet_3d import UCSGNet
from tqdm import tqdm

#Config parameters, set logging
config = Config("3d_config.yml")

model_name = config.model_path.format(config.lr, config.latent_sz, config.num_csg_layer, 
                            config.out_shape_layer, config.batch_size, config.data_size, config.num_dim)
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

train_dataset = ShapeNet_train_dataloader(config)
valid_dataset = ShapeNet_valid_dataloader(config)

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
prev_loss = 100

for epoch in range(config.epochs):
    
    net.train()
    pbar = tqdm(total=len(train_dataset.dataset), leave=False)
    epoch_str = '' if epoch is None else '[Epoch {}/{}]'.format(
            str(epoch).zfill(len(str(config.epochs))), config.epochs)

    train_loss = 0.0
    n = 0.0

    for batch in train_dataset:
        optimizer.zero_grad()
        
        image, pt, dist, bounding_volume = batch
        image = image.to(device)
        pt = pt.to(device)
        dist = dist.to(device)

        pred = net(image, pt)
        pred = pred.squeeze(dim=-1)
        dist = dist.squeeze(dim=-1)

        loss, each_loss = total_loss(pred, dist, net.converter, net.csg_layer, 
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
    n = 0.0
    for batch in valid_dataset:
        with torch.no_grad():
            image, pt, dist, bounding_volume = batch
            image = image.to(device)
            pt = pt.to(device)
            dist = dist.to(device)

            pred = net(image, pt)
            pred = pred.squeeze(dim=-1)
            dist = dist.squeeze(dim=-1)
            loss, each_loss = total_loss(pred, dist, net.converter, net.csg_layer, 
                                            net.evaluator, config.use_planes) 
            valid_loss += loss.item()
            n += 1

        pbar.set_description('{} {} Loss: {:f}'.format(epoch_str, 'Valid', loss.item()))
        pbar.update(image.shape[0])
    
    pbar.close()
    
    mean_valid_loss = valid_loss / n

    log_value('valid_loss', mean_valid_loss, epoch)

    logger.info("Epoch {}/{} => valid_loss: {:f}".format(epoch, config.epochs, mean_valid_loss))
    print("Epoch {}/{} => valid_loss: {:f}".format(epoch, config.epochs, mean_valid_loss))

    if prev_loss > mean_valid_loss:
        logger.info("Saving the Model based on Valid Loss: %f"%(mean_valid_loss))
        print("Saving the Model based on Valid Loss: %f"%(mean_valid_loss), flush=True)
        torch.save(net.state_dict(), "trained_models/{}.pth".format(model_name))
        prev_loss = mean_valid_loss