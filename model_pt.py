import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timeit import default_timer as timer

from dataset_pt import ODT3D_Dataset, ODT3D_Dataset_val
from utils import get_config_from_json, WeightedL1Loss

#from backbone.highresnet import HighRes3DNet
from backbone.unet3d import UNet_ODT


def train_model(config, model_name):
    save_path = os.path.join('./weights', model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Training setting.
    num_epoch = config['epochs']
    save_period = 20
    device = torch.device("cuda:0")
    # Enables benchmark mode in cudnn, which is good whenever your
    # input sizes for your network do not vary.
    torch.backends.cudnn.benchmark = True

    # Prepare data.
    dataset_train = ODT3D_Dataset(config)
    train_gen = DataLoader(dataset_train, batch_size=None, shuffle=False, num_workers=4)
    dataset_val = ODT3D_Dataset_val(config)
    val_gen = DataLoader(dataset_val, batch_size=None, shuffle=False, num_workers=4)

    # Model initialization.
    #model = HighRes3DNet(in_channels=1, out_channels=1)
    model = UNet_ODT()
    model = model.to(device)

    lr = 1e-4
    #loss_fn = nn.SmoothL1Loss()
    loss_fn = WeightedL1Loss(threshold=0.091554, weight=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,
        patience=20, cooldown=5, min_lr=1e-6, eps=1e-08, verbose=True)
    #file_writer = SummaryWriter(log_dir=save_path + '/metrics')
    file_writer = SummaryWriter(log_dir=save_path)

    # Constructs scaler once.
    # GradScaler helps prevent gradients “underflowing”.
    scaler = torch.cuda.amp.GradScaler()

    # Training/val.
    best_val_loss = 99999.
    for epoch in range(1, num_epoch + 1):
        start_time = timer()
        # Reset metrics.
        train_loss = 0.
        val_loss = 0.

        # Training.
        model.train()
        idx_batch = 0
        for X, loss_mask, Y in train_gen:
            # Reduce batchsize to fit VRAM.
            idxset = np.random.choice(20, bz, replace=False)
            X = X[idxset]
            loss_mask = loss_mask[idxset]
            Y = Y[idxset]

            X = X.to(device)
            loss_mask = loss_mask.to(device)
            Y = Y.to(device)

            with torch.cuda.amp.autocast():
                optimizer.zero_grad()
                output = model(X, loss_mask)
                loss = loss_fn(output*loss_mask, Y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
            end_time_trainbatch = timer()
            #print('-->Batch-{:02d}: Training Loss(‰): {:.04f}. \tTime: {:.01f}s'.format(
            #    idx_batch, train_loss/(idx_batch+1)*1000, end_time_trainbatch - start_time))
            idx_batch += 1
        end_time_train = timer()

        # Validation.
        model.eval()
        for X, loss_mask, Y in val_gen:
            X = X[:bz]
            loss_mask = loss_mask[:bz]
            Y = Y[:bz]

            X = X.to(device)
            loss_mask = loss_mask.to(device)
            Y = Y.to(device)

            with torch.cuda.amp.autocast():
                output = model(X, loss_mask)
                loss = loss_fn(output*loss_mask, Y)
                val_loss += loss.item()

        scheduler.step(val_loss)
        train_loss = train_loss/len(train_gen)
        val_loss = val_loss/len(val_gen)
        end_time = timer()
        
        # Display metrics at the end of each epoch. 
        print('Epoch-{:04d}: Training Loss(‰): {:.04f}. \tValidation Loss: {:.04f}. \tTime: {:.01f}/{:.01f}s'.format(
            epoch, train_loss*1000, val_loss*1000, end_time_train - start_time, end_time - start_time))
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # Checkpoint periodly.
        if (epoch % save_period) == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_weights_ep{:03d}.pth'.format(epoch)))
            #model.load_state_dict(torch.load('model_weights.pth'))

        # Checkpoint if improved.
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_weights_best.pth'))
            best_val_loss = val_loss


if __name__ == '__main__':
    #config=get_config_from_json('config.json')
    #train_cl_model()
    #train_cl_triple_model()
    #save_path = r'J:\OCA_inpaint\model_weight\dice_50_softnax'
    #save_path = r'./model_weight/dice_50_softnax_trainval_fixed'
    #save_path = r'J:\RE\OCA_inpainting_re\model_weight\dice_50_softnax_linked_trainval82'
    #train_linked_model(save_path,'dice',config)
    #train_linked_model_v2(save_path,config)

    config = get_config_from_json('config_unet.json')
    print('-----------config-----------\n',config, '\n-----------end-config-----------\n')
    model_name = 'odt3d_unet_bz20_prototype'
    train_model(config, model_name)

