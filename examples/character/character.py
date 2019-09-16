import click
import uuid
import glob
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import Dataset
from shutil import copy2
from torchvision import transforms
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter
from PIL import Image
from ptdec.dec import DEC
from ptdec.model import train, predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy

    
def get_train_compose():
    compose = transforms.Compose([
              transforms.Resize(128),
              transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              transforms.ToTensor(),
              ])
    return compose

def get_test_compose():
    compose = transforms.Compose([
              transforms.Resize(128),
              transforms.ToTensor(),
              ])
    return compose

class Character(Dataset):
    def __init__(self, train, cuda, testing_mode=False):
        super(Character, self).__init__()
        img_dir = './pattern/*.png'
        self.img_files = glob.glob(img_dir)
        self.train = train
        if self.train:
            self.compose = get_train_compose()
        else:
            self.compose = get_test_compose()
        
    def __getitem__(self, index):
        img_filename = self.img_files[index]
        img = Image.open(img_filename)
        img = self.compose(img)
        img = img.view(img.shape[0], -1)
        img = img.squeeze(dim=0)
        print(img.size())
        return img, img_filename
    def __len__(self):
        return 128 if not self.train else len(self.img_files)
        

@click.command()
@click.option(
    '--cuda',
    help='whether to use CUDA (default False).',
    type=bool,
    default=True
)
@click.option(
    '--batch-size',
    help='training batch size (default 256).',
    type=int,
    default=64
)
@click.option(
    '--pretrain-epochs',
    help='number of pretraining epochs (default 300).',
    type=int,
    default=250
)
@click.option(
    '--finetune-epochs',
    help='number of finetune epochs (default 500).',
    type=int,
    default=250
)
@click.option(
    '--testing-mode',
    help='whether to run in testing mode (default False).',
    type=bool,
    default=False
)
def main(
    cuda,
    batch_size,
    pretrain_epochs,
    finetune_epochs,
    testing_mode
):
    writer = SummaryWriter()  # create the TensorBoard object
    # callback function to call during training, uses writer from the scope

    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars('data/autoencoder', {
            'lr': lr,
            'loss': loss,
            'validation_loss': validation_loss,
        }, epoch)
    ds_train = Character(train=True, cuda=cuda, testing_mode=testing_mode)  # training dataset
    ds_val = Character(train=False, cuda=cuda, testing_mode=testing_mode)  # evaluation dataset
    autoencoder = StackedDenoisingAutoEncoder(
        [128*128, 500, 500, 2000, 5],
        final_activation=None
    )
    if cuda:
        autoencoder.cuda()
    print('Pretraining stage.')
    ae.pretrain(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
        corruption=0.2
    )
    print('Training stage.')
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
    ae.train(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
        update_callback=training_callback
    )
    print('DEC stage.')
    model = DEC(
        cluster_number=5,
        hidden_dimension=5,
        encoder=autoencoder.encoder
    )
    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(
        dataset=ds_train,
        model=model,
        epochs=2,
        batch_size=batch_size,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        cuda=cuda
    )
    predicted, actual = predict(ds_train, model, 1, silent=True, return_actual=True, cuda=cuda)
    predicted = predicted.cpu().numpy()
    print(len(predicted), len(actual))
    # copy image file to its cluster class folder.
    def mkdirs(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    for i, p in enumerate(predicted):
        img_dir = './pattern_predicted/{}'.format(p.max())
        mkdirs(img_dir)
        #print(actual[i][0])
        copy2(str(actual[i][0]), img_dir)
        
    #reassignment, accuracy = cluster_accuracy(actual, predicted)
    #print('Final DEC accuracy: %s' % accuracy)
#     if not testing_mode:
#         predicted_reassigned = [reassignment[item] for item in predicted]  # TODO numpify
#         confusion = confusion_matrix(actual, predicted_reassigned)
#         normalised_confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
#         confusion_id = uuid.uuid4().hex
#         sns.heatmap(normalised_confusion).get_figure().savefig('confusion_%s.png' % confusion_id)
#         print('Writing out confusion diagram with UUID: %s' % confusion_id)
#         writer.close()


if __name__ == '__main__':
    main()
