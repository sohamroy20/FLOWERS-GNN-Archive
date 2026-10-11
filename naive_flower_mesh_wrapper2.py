import torch
import torch.nn as nn

from flowers.models.flower import Flower


class NaiveFlowerMeshWrapper2(nn.Module):
    """
    Version 2 MGN-aligned naive FLOWERS mesh baseline.

    Forward path:
        mesh velocity
        -> mesh_to_grid
        -> add cylinder mask channel
        -> smaller FLOWERS model
        -> grid_to_mesh using torch grid_sample
        -> predicted mesh velocity delta

    This version keeps the same task as MGN:
        input  = vx, vy
        output = delta_vx, delta_vy
    """

    def __init__(
        self,
        bridge,
        in_mesh_channels=2,
        out_mesh_channels=2,
        add_mask_channel=True,
    ):
        super().__init__()

        self.bridge = bridge
        self.add_mask_channel = add_mask_channel

        in_grid_channels = in_mesh_channels + (1 if add_mask_channel else 0)

        self.base_model = Flower(
            dim_in=in_grid_channels,
            dim_out=out_mesh_channels,
            n_spatial_dims=2,
            spatial_resolution=(bridge.ny, bridge.nx),

            # Stability improvement:
            # smaller model than default FLOWERS.
            lifting_dim=96,
            num_heads=32,
            groups=32,
        )

    def forward(self, mesh_velocity):
        """
        Args:
            mesh_velocity: (B, N, 2)

        Returns:
            mesh_delta_velocity: (B, N, 2)
        """

        grid = self.bridge.mesh_to_grid(mesh_velocity)

        if self.add_mask_channel:
            B = grid.shape[0]
            mask = self.bridge.mask_torch.expand(B, -1, -1, -1)
            grid = torch.cat([grid, mask], dim=1)

        grid_delta = self.base_model(grid)

        mesh_delta = self.bridge.grid_to_mesh(grid_delta)

        return mesh_delta