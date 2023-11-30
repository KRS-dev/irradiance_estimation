from neuralop import FNO2d, FNO



class FNO2d_estimation(FNO):

    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        hidden_channels,
        in_channels=11,
        out_channels=1,
        lifting_channels=32,
        projection_channels=32,
        n_layers=4,
        max_n_modes=None,
        fno_block_precision="full",
        stabilizer=None,
        norm=None,
        fno_skip="linear",
        mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,
            norm=norm,
            fno_skip=fno_skip,
            mlp_skip=mlp_skip,
            separable=separable,
            factorization=factorization,
            **kwargs
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width

    
