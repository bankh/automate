import torch
import torch.nn as nn
import pytorch_lightning as pl
from automate import SBGCN, PartDataset, run_model, ArgparseInitialized, PointNetEncoder

class CADDecoder(nn.Module):
    """
    A neural network module for decoding CAD geometry.
    
    This decoder takes a latent representation of CAD geometry 
    (typically produced by an encoder) and UV coordinates as input, 
    and outputs reconstructed 3D points and an additional feature.
    The architecture uses skip connections to preserve information 
    from the input latent vector throughout the decoding process.
    """
    def __init__(self, input_dim, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.ReLu(),
            nn.Linear(hidden_dim + input_dim + 2, hidden_dim),
            nn.ReLu(),
            nn.Linear(hidden_dim + input_dim + 2, hidden_dim),
            nn.ReLu(),
            nn.Linear(hidden_dim + input_dim + 2, 4)
        )
    def forward(self, z, uv):
        x = torch.cat([z, uv], dim=-1)
        for layer in self.net[:-1]:
            if isinstance(layer, nn.Linear):
                x = torch.cat([x, z, uv], dim=-1)
            x = layer(x)
        return self.net[-1](torch.cat([x, z, uv], dim=-1))

class CADAutoencoder(pl.LightningModule,ArgparseInitialized):
    """
    A PyTorch Lightning module for training a CAD autoencoder.
    
    This class defines a CAD autoencoder model using SBGCN for encoding 
    and a custom decoder for reconstruction.
    It also includes a PointNet encoder for feature extraction from the 
    ground truth points.
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.encoder = SBGCN(hparams)
        self.encoder = SBGCN(
            f_in_width=self.hparams.f_in_width,
            l_in_width=self.hparams.l_in_width,
            e_in_width=self.hparams.e_in_width,
            v_in_width=self.hparams.v_in_width,
            out_width=self.hparams.out_width,
            k=self.hparams.k,
            use_uvnet_features=self.hparams.use_uvnet_features,
        )
        self.decoder = CADDecoder(self.hparams.out_width)
        self.pointnet = PointNetEncoder(K=3, 
                                        layers=(64,64,64,128, self.hparams.out_width))

    def forward(self, batch):
        x_t, x_p, x_f, x_l, x_e, x_v = self.encoder(batch)
        reconstructed = self.decoder(x_t, batch.uv)
        return reconstructed
    
    def training_step(self, batch, batch_idx):
        x_t, x_p, x_f, x_l, x_e, x_v = self.encoder(batch)
        reconstructed = self.decoder(x_t, batch.uv)

        # Get around truth points (needs adjustment)
        gt_points = batch.face_samples[:, :3, :, :].reshape(x_f.shape[0], -1, 3)
        
        # Compute reconstruction loss
        recon_loss = nn.MSELoss()(reconstructed[:,:3], gt_points)

        # Compute pointnet loss
        _, x_p_pointnet = self.pointnet(gt_points)
        pointnet_loss = nn.MSELoss()(x_p, x_p_pointnet)

        # Total loss
        loss = recon_loss + pointnet_loss

        self.log('train_loss', loss)
        self.log('recon_loss', recon_loss)
        self.log('pointnet_loss', pointnet_loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 
                                lr=self.hparams.learning_rate)

    
@staticmethod
def add_model_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("CADAutoencoder")
    parser.add_argument('--f in_width', type=int, default=56)
    parser.add_argument('--l in_width', type=int, default=38)
    parser.add_argument('--e in_width', type=int, default=63)
    parser.add_argument('--v in_width', type=int, default=3)
    parser.add_argument('--out_width', type=int, default=64)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--use_uvnet_features', type=bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=0.0005)


if __name__ == '__main__':
    run_model({
        'model_class': 'CADAutoencoder',
        'data_class': 'PartDataset',
        'gpus': 1,
        'max_epochs': 100,
        })