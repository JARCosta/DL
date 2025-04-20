# hw2/data.py

import functools
import os
from typing import Callable, Dict, List, Tuple

from ml_collections import config_dict
import torch
import torchaudio
import torchvision


class TokenVectorizer(torch.nn.Module):
    def __init__(
        self,
        n_vocab: int,
        max_len: int = 50,
        start: str = "<",
        stop: str = ">",
        empty: str = "@",
        unk: str = "#",
    ) -> None:
        super().__init__()
        self._max_len = max_len
        self._start = start
        self._stop = stop
        self._empty = empty
        self._unk = unk
        self._vocab = [" ", ",", ".", "?", "-"]
        self._vocab += [chr(i + 96) for i in range(1, 27)]

        assert self._start not in self._vocab
        assert self._stop not in self._vocab
        assert self._empty not in self._vocab
        assert self._unk not in self._vocab
        self._vocab = [self._start, self._stop, self._empty, self._unk] + self._vocab
        assert len(self._vocab) == n_vocab
        self._unk_idx = 3
        self._mapper = {c: i for i, c in enumerate(self._vocab)}
        self._inv_mapper = {v: k for k, v in self._mapper.items()}

    def forward(self, transcript: str) -> torch.Tensor:
        transcript = transcript.lower()[: self._max_len - 2]
        transcript = self._start + transcript + self._stop
        pad = [self._mapper[self._empty]] * (self._max_len - len(transcript))
        return torch.tensor(
            [self._mapper.get(c, self._unk_idx) for c in transcript] + pad
        )

    @property
    def invert_mapper(self) -> Dict[int, str]:
        return self._inv_mapper

    @property
    def mapper(self) -> Dict[str, int]:
        return self._mapper

    @property
    def vocab(self) -> List[str]:
        return self._vocab


class SpeechRecognitionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds: torch.utils.data.Dataset,
        get_el: Callable,
        audio_transforms: torch.nn.Module,
        text_transforms: torch.nn.Module,
    ):
        self._orig_ds = ds
        self._get_el = get_el
        self._audio_tf = audio_transforms
        self._text_tf = text_transforms

    def __getitem__(self, idx):
        audio, text = self._get_el(self._orig_ds, idx)
        audio = self._audio_tf(audio)
        text = self._text_tf(text)
        return audio, text

    def __len__(self):
        return len(self._orig_ds)


def _get_ljspeech_el(
    ds: torch.utils.data.Dataset, idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    el = ds[idx]
    return el[0], el[-1]


def _log_mel_stft_transforms(config: config_dict.ConfigDict) -> torch.nn.Module:
    def log_mel(x):
        log_x = torch.clamp(x, min=1e-10).log10()
        log_x = torch.maximum(log_x, log_x.max() - 8.0)
        log_x = (log_x + 4.0) / 4.0
        return log_x

    return torchvision.transforms.Compose(
        [
            functools.partial(torch.squeeze, dim=0),
            torchaudio.transforms.Spectrogram(**config.spectrogram),
            torchaudio.transforms.MelScale(**config.mel_scale),
            log_mel,
            torchvision.transforms.Pad(
                (config.pad.l, config.pad.t, config.pad.r, config.pad.b)
            ),
            lambda x: x[..., :, : config.pad.r],
        ]
    )


def build_dl(
    split: DataSplit, config: config_dict.ConfigDict
) -> torch.utils.data.DataLoader:
    os.makedirs(config.data_folder, exist_ok=True)
    orig_ds = torchaudio.datasets.LJSPEECH(config.data_folder, download=True)
    get_el = _get_ljspeech_el
    audio_tf = _log_mel_stft_transforms(config.transforms.audio)
    text_tf = TokenVectorizer(**config.transforms.text.token_vectorizer)

    ds = torch.utils.data.Subset(orig_ds, range(*config.data_split.get(split)))
    ds = SpeechRecognitionDataset(ds, get_el, audio_tf, text_tf)
    dl = torch.utils.data.DataLoader(ds, **config.dataloader.get(split))
    return dl

