

class TrainingOptions:
    """
    Configuration options for the training
    """

    def __init__(self,
                 batch_size: int,
                 number_of_epochs: int,
                 train_folder: str, validation_folder: str, runs_folder: str,
                 start_epoch: int, experiment_name: str, using_new_model: bool = False):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.runs_folder = runs_folder
        self.start_epoch = start_epoch
        self.experiment_name = experiment_name
        if using_new_model:
            self.experiment_name = 'new_model_' + self.experiment_name


class BasicConfiguration():
    """
    The Basic network configuration.
    """

    def __init__(self, H: int, W: int, message_length: int,
                 encoder_blocks: int, encoder_channels: int,
                 decoder_blocks: int, decoder_channels: int,
                 use_discriminator: bool,
                 use_vgg: bool,
                 discriminator_blocks: int, discriminator_channels: int,
                 decoder_loss: float,
                 encoder_loss: float,
                 spa_decoder_loss: float,
                 dwt_decoder_loss: float,
                 dct_decoder_loss: float,
                 svd_decoder_loss: float,
                 adversarial_loss: float,
                 enable_fp16: bool = False, using_new_model: bool = False, using_res_model: bool = False,
                 using_attention: bool = False, using_sp_attention: bool = False, dl_res_attention: bool = False):
        self.H = H
        self.W = W
        self.message_length = message_length
        self.encoder_blocks = encoder_blocks
        self.encoder_channels = encoder_channels
        self.use_discriminator = use_discriminator
        self.use_vgg = use_vgg
        self.decoder_blocks = decoder_blocks
        self.decoder_channels = decoder_channels
        self.discriminator_blocks = discriminator_blocks
        self.discriminator_channels = discriminator_channels
        self.decoder_loss = decoder_loss
        self.encoder_loss = encoder_loss
        # spa_decoded_messages, dwt_decoded_messages, dct_decoded_messages, svd_decoded_messages
        self.spa_decoder_loss = spa_decoder_loss
        self.dwt_decoder_loss = dwt_decoder_loss
        self.dct_decoder_loss = dct_decoder_loss
        self.svd_decoder_loss = svd_decoder_loss

        self.adversarial_loss = adversarial_loss
        self.enable_fp16 = enable_fp16
        self.using_new_model = using_new_model
        self.using_res_model = using_res_model
        self.using_attention = using_attention
        self.using_sp_attention = using_sp_attention

        self.dl_res_attention = dl_res_attention


def getDefaultHiddenConfig(H, W, message_length, model_name: str = "HiDDeN"):

    newmodel = False
    resmodel = False
    attention = False
    sp_attention = False
    dl_res_attention = False

    hidden_config = BasicConfiguration(H=H, W=W,
                                        message_length=message_length,
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=7, decoder_channels=64,
                                        use_discriminator=True,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        encoder_loss=2,
                                        decoder_loss=1.2,
                                        spa_decoder_loss=0.3,
                                        dwt_decoder_loss=0.3,
                                        dct_decoder_loss=0.4,
                                        svd_decoder_loss=0.5,
                                        adversarial_loss=1e-3,
                                        enable_fp16=False,
                                        using_new_model=newmodel,
                                        using_res_model=resmodel,
                                        using_attention=attention,
                                        using_sp_attention=sp_attention,
                                        dl_res_attention=dl_res_attention
                                        )
    return hidden_config

def get_AEGISnoAttention_noRes_Config(H, W, message_length, model_name: str = "AEGISnoAttention_noRes"):

    attention = False
    sp_attention = False
    dl_res_attention = False

    newmodel = True
    resmodel = False

    AEGISnoAttention_noRes_config = BasicConfiguration(H=H, W=W,
                                        message_length=message_length,
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=7, decoder_channels=64,
                                        use_discriminator=True,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        encoder_loss=2,
                                        decoder_loss=1.2,
                                        spa_decoder_loss=0.3,
                                        dwt_decoder_loss=0.3,
                                        dct_decoder_loss=0.4,
                                        svd_decoder_loss=0.5,
                                        adversarial_loss=1e-3,
                                        enable_fp16=False,
                                        using_new_model=newmodel,
                                        using_res_model=resmodel,
                                        using_attention=attention,
                                        using_sp_attention=sp_attention,
                                        dl_res_attention=dl_res_attention
                                        )
    return AEGISnoAttention_noRes_config


def get_AEGISnoAttentionConfig(H, W, message_length, model_name: str = "AEGISnoAttention"):

    attention = False
    sp_attention = False
    dl_res_attention = False

    newmodel = True
    resmodel = True

    AEGISnoAttention_config = BasicConfiguration(H=H, W=W,
                                        message_length=message_length,
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=7, decoder_channels=64,
                                        use_discriminator=True,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        encoder_loss=2,
                                        decoder_loss=1.2,
                                        spa_decoder_loss=0.3,
                                        dwt_decoder_loss=0.3,
                                        dct_decoder_loss=0.4,
                                        svd_decoder_loss=0.5,
                                        adversarial_loss=1e-3,
                                        enable_fp16=False,
                                        using_new_model=newmodel,
                                        using_res_model=resmodel,
                                        using_attention=attention,
                                        using_sp_attention=sp_attention,
                                        dl_res_attention=dl_res_attention
                                        )
    return AEGISnoAttention_config

def get_SP_AEGIS_Config(H, W, message_length, model_name: str = "SP_AEGIS"):

    dl_res_attention = False

    newmodel = True
    resmodel = True
    attention = True
    sp_attention = True

    SP_AEGIS_config = BasicConfiguration(H=H, W=W,
                                        message_length=message_length,
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=7, decoder_channels=64,
                                        use_discriminator=True,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        encoder_loss=2,
                                        decoder_loss=1.2,
                                        spa_decoder_loss=0.3,
                                        dwt_decoder_loss=0.3,
                                        dct_decoder_loss=0.4,
                                        svd_decoder_loss=0.5,
                                        adversarial_loss=1e-3,
                                        enable_fp16=False,
                                        using_new_model=newmodel,
                                        using_res_model=resmodel,
                                        using_attention=attention,
                                        using_sp_attention=sp_attention,
                                        dl_res_attention=dl_res_attention
                                        )
    return SP_AEGIS_config

def get_GL_AEGIS_Config(H, W, message_length, model_name: str = "GL_AEGIS"):

    dl_res_attention = False

    newmodel = True
    resmodel = True
    attention = True
    sp_attention = False

    GL_AEGIS_config = BasicConfiguration(H=H, W=W,
                                        message_length=message_length,
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=7, decoder_channels=64,
                                        use_discriminator=True,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        encoder_loss=2,
                                        decoder_loss=1.2,
                                        spa_decoder_loss=0.3,
                                        dwt_decoder_loss=0.3,
                                        dct_decoder_loss=0.4,
                                        svd_decoder_loss=0.5,
                                        adversarial_loss=1e-3,
                                        enable_fp16=False,
                                        using_new_model=newmodel,
                                        using_res_model=resmodel,
                                        using_attention=attention,
                                        using_sp_attention=sp_attention,
                                        dl_res_attention=dl_res_attention
                                        )
    return GL_AEGIS_config


def get_DL_AEGIS_Config(H, W, message_length, model_name: str = "DL_AEGIS"):

    newmodel = True
    resmodel = True
    attention = True
    sp_attention = True
    dl_res_attention = True

    DL_AEGIS_config = BasicConfiguration(H=H, W=W,
                                        message_length=message_length,
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=7, decoder_channels=64,
                                        use_discriminator=True,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        encoder_loss=2,
                                        decoder_loss=1.2,
                                        spa_decoder_loss=0.3,
                                        dwt_decoder_loss=0.3,
                                        dct_decoder_loss=0.4,
                                        svd_decoder_loss=0.5,
                                        adversarial_loss=1e-3,
                                        enable_fp16=False,
                                        using_new_model=newmodel,
                                        using_res_model=resmodel,
                                        using_attention=attention,
                                        using_sp_attention=sp_attention,
                                        dl_res_attention=dl_res_attention
                                        )
    return DL_AEGIS_config