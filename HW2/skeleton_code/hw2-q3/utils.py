# hw2/utils.py

import enum
import pickle

import matplotlib.pyplot as plt
import numpy as np


def save_obj(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_txt(str, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(str)


def load_metrics(file_path):
    with open(file_path, "rb") as f:
        metrics = pickle.load(f)

    return metrics


def plot_metrics(metrics, plot_path, phase="train"):
    metric_keys = list(metrics[0][phase].keys())
    len_x = len(metrics)
    x_label = "epoch".title()
    x = np.arange(len_x)

    for k in metric_keys:
        if k == "n":
            continue

        metric_path = plot_path / f"{k}__{phase}.pdf"
        y = np.array([metrics[i][phase][k] for i in range(len_x)])

        if k != "loss":
            batch = np.array([metrics[i][phase]["n"] for i in range(len_x)])
            y /= batch

        y = y.mean(axis=-1)
        _create_plot([y], [x], x_label, k.title(), metric_path, show=True)


def cmp_phase_metrics(metrics, key, phases, plot_path, labels):
    len_x = len(metrics)
    x_label = "epoch".title()
    x = np.arange(len_x)
    y = []

    for phase in phases:
        y_phase = np.array([metrics[i][phase][key] for i in range(len_x)])

        if key != "loss":
            batch = np.array([metrics[i][phase]["n"] for i in range(len_x)])
            y_phase /= batch

        y_phase = y_phase.mean(axis=-1)
        y.append(y_phase)

    x = [x] * len(y)
    fig_path = plot_path / f"{key}__{'_'.join(phases)}.pdf"
    labels = [labels[p] for p in phases]
    _create_plot(y, x, x_label, key.title(), fig_path, line_labels=labels, show=True)


def _create_plot(ys, xs, x_label, y_label, save_path, line_labels=None, show=False):
    fig, ax = plt.subplots()
    for y, x in zip(ys, xs):
        ax.plot(x, y)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")

    if line_labels is not None:
        ax.legend(line_labels)

    if show:
        plt.show(fig)

    plt.close(fig)


class Dataset(str, enum.Enum):
    LJSPEECH_STFT = "ljspeech_stft"
    LJSPEECH_MEL = "ljspeech_mel"


class DataSplit(str, enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class AudioTransformType(str, enum.Enum):
    STFT = "stft"
    LOG_MEL_STFT = "log_mel_stft"


class EncoderType(str, enum.Enum):
    RNN = "rnn"
    TRANSFORMER = "transformer"


class DecoderType(str, enum.Enum):
    RNN = "rnn"
    TRANSFORMER = "transformer"
