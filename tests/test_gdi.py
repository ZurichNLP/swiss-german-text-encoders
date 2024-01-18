from unittest import TestCase

from evaluation.gdi.gdi_datasets import GDIVardial2019Dataset


class GDIVardial2019DatasetTestCase(TestCase):

    def setUp(self):
        self.dataset = GDIVardial2019Dataset()

    def test_num_samples(self):
        dataset = self.dataset.dataset
        self.assertEqual(len(dataset['train']), 14279)
        self.assertEqual(len(dataset['validation']), 4530)
        self.assertEqual(len(dataset['test']), 4743)

    def test_sample(self):
        sample = self.dataset.dataset['train'][0]
        self.assertEqual(sample['text'], 'under em missbrüüchliche drugg vo der ijuschtiz natürlig in däm fall he')
        self.assertEqual(sample['label'], 'BS')
        self.assertEqual(sample['lang'], 'gsw')

    def test_labels(self):
        labels = ['BS', 'BE', 'LU', 'ZH']
        for split in ['train', 'validation', 'test']:
            for label in labels:
                self.assertIn(label, self.dataset.dataset[split]['label'])

    def test_print_statistics(self):
        self.dataset.print_statistics()

    def test_required_seq_len(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        max_seq_len = 0
        for split in ['train', 'validation', 'test']:
            for sample in self.dataset.dataset[split]:
                max_seq_len = max(max_seq_len, len(tokenizer.tokenize(sample['text'])))
        print(max_seq_len)
