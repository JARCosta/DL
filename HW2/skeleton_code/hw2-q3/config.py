# hw2/config.py

from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.seed = 42
    config.n_fft = 400
    config.n_vocab = 35
    config.text_len = 99
    config.audio_freq = 80  # number of mel filters
    config.audio_time = 2754  # 10 seconds of sound
    config.epochs = 15
    config.lr = 1e-3
    config.unk_token_idx = 3
    config.storage_folder = "./storage"
    config.speech_to_text = dict(
        max_text_len=config.get_ref("text_len"),
        encoder_kwargs=dict(
            n_layers=2,
            freq_dim=config.get_ref("audio_freq"),
            time_dim=config.get_ref("audio_time") // 2,
            hidden_dim=200,
            n_heads=2,
            feed_fwd_dim=400,
        ),
        decoder_type=DecoderType.TRANSFORMER,
        decoder_kwargs=dict(
            freq_dim=config.get_ref("n_vocab"),
            time_dim=config.get_ref("text_len"),
            hidden_dim=200,
            n_heads=2,
            feed_fwd_dim=400,
        ),
    )

    config.data = dict(
        data_folder="./data/",
        data_split=dict(train=(0, 9825), validation=(9825, 10480), test=(10480, 13100)),
        dataloader=dict(
            train=dict(batch_size=64, shuffle=True, num_workers=2, prefetch_factor=2),
            validation=dict(
                batch_size=16, shuffle=False, num_workers=2, prefetch_factor=2
            ),
            test=dict(
                batch_size=16, shuffle=False, num_workers=2, prefetch_factor=2
            ),
        ),
        transforms=dict(
            audio=dict(
                spectrogram=dict(
                    n_fft=config.get_ref("n_fft"),
                    hop_length=160,
                    power=2,
                    center=True,
                    normalized=False,
                    onesided=None,
                ),
                mel_scale=dict(
                    n_mels=config.get_ref("audio_freq"),
                    n_stft=config.get_ref("n_fft") // 2 + 1,
                    norm="slaney",
                    mel_scale="slaney",
                ),
                pad=dict(
                    l=0,
                    t=0,
                    r=config.get_ref("audio_time"),
                    b=0,
                ),
                freq_masking=dict(freq_mask_param=27),
                time_masking=dict(time_mask_param=80),
            ),
            text=dict(
                token_vectorizer=dict(
                    max_len=config.get_ref("text_len"),
                    start="<",
                    stop=">",
                    empty="@",
                    unk="#",
                    n_vocab=config.get_ref("n_vocab"),
                )
            ),
        ),
    )

    return config.lock()
