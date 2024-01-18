from unittest import TestCase

from evaluation.pos.pos_datasets import StandardGermanUDDataset, SwissGermanPOSDataset


class StandardGermanUDDatasetTestCase(TestCase):

    def setUp(self):
        self.dataset = StandardGermanUDDataset()

    def test_sample(self):
        sample = self.dataset.dataset["train"][0]
        self.assertEqual(sample["tokens"], ['Hinter', 'der', 'neuen', 'Firma', 'steht', 'unter', 'anderem', 'Lucent', 'Technologies', ',', 'einer', 'der', 'größten', 'Anbieter', 'von', 'Equipment', 'für', 'Netzwerke', 'und', 'Telekommunikation', '.'])
        self.assertEqual(sample["upos"], ['ADP', 'DET', 'ADJ', 'NOUN', 'VERB', 'ADP', 'DET', 'PROPN', 'X', 'PUNCT', 'DET', 'DET', 'ADJ', 'NOUN', 'ADP', 'NOUN', 'ADP', 'NOUN', 'CCONJ', 'NOUN', 'PUNCT'])
        self.assertEqual(sample["stts"], ['APPR', 'ART', 'ADJA', 'NN', 'VVFIN', 'APPR', 'PIS', 'NE', 'FM', '$,', 'PIS', 'ART', 'ADJA', 'NN', 'APPR', 'NN', 'APPR', 'NN', 'KON', 'NN', '$.'])
        self.assertEqual(sample["lang"], 'de')
        print(sample)

    def test_num_samples(self):
        dataset = self.dataset.dataset
        self.assertEqual(len(dataset["train"]), 75617)
        self.assertEqual(len(dataset["validation"]), 18434)
        self.assertEqual(len(dataset["test"]), 18459)

    def test_print_statistics(self):
        self.dataset.print_statistics()


class SwissGermanPOSDatasetTestCase(TestCase):

    def setUp(self):
        self.dataset = SwissGermanPOSDataset()

    def test_sample(self):
        sample = self.dataset.dataset["test"][0]
        self.assertEqual(sample["tokens"], ['Mit', 'de', 'Eroberig', 'vom', 'Aargau', 'durch', 'di', 'alti', 'Eidgnosseschaft', 'im', '1415i', 'isch', 'Bade', 'de', 'Sitz', 'vom', 'Landvogt', 'vo', 'de', 'Grafschaft', 'Bade', 'worde', 'und', 'au', 'vili', 'Tagsatzige', 'hei', 'hiir', 'schtattgfunde', '.'])
        self.assertEqual(sample["upos"], ['ADP', 'DET', 'NOUN', 'ADP', 'PROPN', 'ADP', 'DET', 'ADJ', 'NOUN', 'ADP', 'NUM', 'AUX', 'PROPN', 'DET', 'NOUN', 'ADP', 'NOUN', 'ADP', 'DET', 'NOUN', 'PROPN', 'AUX', 'CCONJ', 'ADV', 'DET', 'NOUN', 'AUX', 'ADV', 'VERB', 'PUNCT'])
        self.assertEqual(sample["lang"], 'gsw')

    def test_num_samples(self):
        dataset = self.dataset.dataset
        self.assertEqual(len(dataset["test"]), 7320)

    def test_print_statistics(self):
        self.dataset.print_statistics()
