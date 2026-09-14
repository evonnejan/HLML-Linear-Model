"""Check evaluation coverage with a real DataLoader; no data files or training."""
from types import SimpleNamespace

import pytest
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

import data_provider.Data_Factory as factory


@pytest.fixture
def make_loader(monkeypatch):
    def build(flag, n, batch_size=4):
        class IndexedDataset(Dataset):
            def __init__(self, **kwargs):
                self.options = kwargs

            def __len__(self):
                return n

            def __getitem__(self, index):
                return index

        monkeypatch.setitem(factory.data_dict, "custom", IndexedDataset)
        monkeypatch.setattr(factory, "Dataset_Custom", IndexedDataset)
        # Separate type preserves the factory's prediction path.
        class PredictionDataset(IndexedDataset):
            pass
        monkeypatch.setattr(factory, "Dataset_Pred", PredictionDataset)
        args = SimpleNamespace(
            data="custom", embed="timeF", batch_size=batch_size, freq="min",
            stride_train=3, stride_eval=1, root_path="unused", data_path="unused.csv",
            seq_len=60, label_len=30, pred_len=15, features="S", target="HL01",
            input_col="HL02", exog_col=None, segment_col="segment_id",
            model="DLinearMix2", split_file="unused-split.csv", fold=1, num_workers=0,
        )
        return factory.data_provider(args, flag)
    return build


@pytest.mark.parametrize("n", [1, 3, 4, 5, 11])
def test_validation_visits_every_window_in_order_on_every_pass(make_loader, n):
    dataset, loader = make_loader("val", n)
    assert dataset.options["stride"] == 1
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.drop_last is False
    for _ in range(3):
        batches = [batch.tolist() for batch in loader]
        assert [i for batch in batches for i in batch] == list(range(n))
        assert len(batches[-1]) == (n % 4 or 4)


def test_training_keeps_random_sampling_and_full_batches(make_loader):
    dataset, loader = make_loader("train", 11)
    assert dataset.options["stride"] == 3
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.drop_last is True
    batches = [batch.tolist() for batch in loader]
    assert [len(batch) for batch in batches] == [4, 4]
    assert len(set(i for batch in batches for i in batch)) == 8


def test_test_loader_keeps_partial_batch(make_loader):
    _, loader = make_loader("test", 5)
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.drop_last is False
    assert [batch.tolist() for batch in loader] == [[0, 1, 2, 3], [4]]


def test_prediction_loader_keeps_batch_size_one(make_loader):
    _, loader = make_loader("pred", 3)
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.drop_last is False
    assert loader.batch_size == 1
    assert [batch.tolist() for batch in loader] == [[0], [1], [2]]
