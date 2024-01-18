from pathlib import Path

from pretraining.pretraining_datasets import AdaptationDataset

data_dir = Path(__file__).parent.parent / 'data'
assert data_dir.exists()

save_dir = data_dir / 'continued_pretraining'

dataset = AdaptationDataset()
dataset.dataset.save_to_disk(save_dir)
