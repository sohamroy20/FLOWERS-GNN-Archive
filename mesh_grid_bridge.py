import numpy as np
import torch
import matplotlib.tri as mtri


class StaticMeshGridBridge:
    """
    Class to convert between mesh and grid representations for 2D data.
    The mesh is defined by node positions and triangular cells.
    The grid is a regular 2D grid covering the bounding box of the mesh.
     - mesh_to_grid uses barycentric interpolation based on the triangle each grid point falls into
     - grid_to_mesh uses bilinear interpolation based on the 4 nearest grid points to each mesh node
     - A mask is created to identify grid points that fall inside a specified cylindrical region
    Precompute:
      - mesh -> grid interpolation using barycentric weights on triangles
      - grid -> mesh interpolation using bilinear weights on regular grid
    """

    def __init__(
        self,
        mesh_pos: np.ndarray,   # (N, 2)
        cells: np.ndarray,      # (Tcells, 3)
        nx: int,
        ny: int,
        cyl_center=(0.3, 0.125),
        cyl_radius=0.05,
        device="cpu",
    ):
        self.mesh_pos = mesh_pos
        self.cells = cells
        self.nx = nx
        self.ny = ny
        self.device = device

        self.x_min = float(mesh_pos[:, 0].min())
        self.x_max = float(mesh_pos[:, 0].max())
        self.y_min = float(mesh_pos[:, 1].min())
        self.y_max = float(mesh_pos[:, 1].max())

        xs = np.linspace(self.x_min, self.x_max, nx)
        ys = np.linspace(self.y_min, self.y_max, ny)
        grid_x, grid_y = np.meshgrid(xs, ys)

        self.grid_x = grid_x.astype(np.float32)
        self.grid_y = grid_y.astype(np.float32)

        cx, cy = cyl_center
        r = cyl_radius
        self.mask = (((grid_x - cx) ** 2 + (grid_y - cy) ** 2) <= r ** 2).astype(np.float32)
        self.mask_torch = torch.tensor(self.mask[None, None, ...], dtype=torch.float32, device=device)

        self.M2G = self._build_mesh_to_grid_matrix()   # (G, N)
        self.G2M = self._build_grid_to_mesh_matrix()   # (N, G)

        self.M2G = torch.tensor(self.M2G, dtype=torch.float32, device=device)
        self.G2M = torch.tensor(self.G2M, dtype=torch.float32, device=device)

    def _build_mesh_to_grid_matrix(self):
        pos = self.mesh_pos
        tri = mtri.Triangulation(pos[:, 0], pos[:, 1], self.cells)
        trifinder = tri.get_trifinder()

        gx = self.grid_x.ravel()
        gy = self.grid_y.ravel()
        G = len(gx)
        N = len(pos)

        A = np.zeros((G, N), dtype=np.float32)
        tri_ids = trifinder(gx, gy)

        for i in range(G):
            if self.mask.ravel()[i] > 0.5:
                continue

            tid = tri_ids[i]
            px, py = gx[i], gy[i]

            if tid == -1:
                d2 = (pos[:, 0] - px) ** 2 + (pos[:, 1] - py) ** 2
                j = np.argmin(d2)
                A[i, j] = 1.0
                continue

            ids = self.cells[tid]
            p0, p1, p2 = pos[ids[0]], pos[ids[1]], pos[ids[2]]

            T = np.array([
                [p0[0], p1[0], p2[0]],
                [p0[1], p1[1], p2[1]],
                [1.0,   1.0,   1.0],
            ], dtype=np.float64)
            b = np.array([px, py, 1.0], dtype=np.float64)

            try:
                w = np.linalg.solve(T, b)
                A[i, ids[0]] = w[0]
                A[i, ids[1]] = w[1]
                A[i, ids[2]] = w[2]
            except np.linalg.LinAlgError:
                d2 = (pos[:, 0] - px) ** 2 + (pos[:, 1] - py) ** 2
                j = np.argmin(d2)
                A[i, j] = 1.0

        return A

    def _build_grid_to_mesh_matrix(self):
        pos = self.mesh_pos
        N = len(pos)
        G = self.nx * self.ny
        B = np.zeros((N, G), dtype=np.float32)

        xs = np.linspace(self.x_min, self.x_max, self.nx)
        ys = np.linspace(self.y_min, self.y_max, self.ny)

        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]

        for i, (x, y) in enumerate(pos):
            fx = (x - self.x_min) / dx
            fy = (y - self.y_min) / dy

            ix0 = int(np.floor(fx))
            iy0 = int(np.floor(fy))
            ix1 = min(ix0 + 1, self.nx - 1)
            iy1 = min(iy0 + 1, self.ny - 1)

            ix0 = max(ix0, 0)
            iy0 = max(iy0, 0)

            tx = fx - ix0
            ty = fy - iy0

            w00 = (1 - tx) * (1 - ty)
            w10 = tx * (1 - ty)
            w01 = (1 - tx) * ty
            w11 = tx * ty

            def flat_idx(iy, ix):
                return iy * self.nx + ix

            B[i, flat_idx(iy0, ix0)] += w00
            B[i, flat_idx(iy0, ix1)] += w10
            B[i, flat_idx(iy1, ix0)] += w01
            B[i, flat_idx(iy1, ix1)] += w11

        return B

    def mesh_to_grid(self, mesh_feats: torch.Tensor):
        """
        mesh_feats: (B, N, C)
        returns: (B, C, H, W)
        """
        B, N, C = mesh_feats.shape
        grid_flat = torch.einsum("gn,bnc->bgc", self.M2G, mesh_feats)
        grid = grid_flat.view(B, self.ny, self.nx, C).permute(0, 3, 1, 2)
        return grid

    def grid_to_mesh(self, grid_feats: torch.Tensor):
        """
        grid_feats: (B, C, H, W)
        returns: (B, N, C)
        """
        B, C, H, W = grid_feats.shape
        grid_flat = grid_feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
        mesh = torch.einsum("ng,bgc->bnc", self.G2M, grid_flat)
        return mesh