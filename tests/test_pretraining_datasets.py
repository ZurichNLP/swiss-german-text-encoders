from pathlib import Path
from unittest import TestCase

from datasets import load_from_disk

from pretraining.pretraining_datasets import SwissCrawlDataset, SwissGermanTweetsDataset, SwissdoxDataset, AdaptationDataset


class SwissCrawlDatasetTestCase(TestCase):

        def setUp(self):
            self.dataset = SwissCrawlDataset()

        def test_sample(self):
            sample = self.dataset.dataset['train'][0]
            self.assertEqual(sample["text"], "Me het dä Chram zsämepackt, und der Papa het wyters nid gschmählt, wil ihm dem Töldi sy Streich der Pfarrer vo Dießbach in Erinnerung bracht het.")
            self.assertEqual(sample["lang"], "gsw")

        def test_num_samples(self):
            self.assertEqual(len(self.dataset.dataset['train']), 563037)

        def test_print_statistics(self):
            self.dataset.print_statistics()


class SwissGermanTweetsDatasetTestCase(TestCase):

            def setUp(self):
                self.dataset = SwissGermanTweetsDataset()

            def test_sample(self):
                sample = self.dataset.dataset["train"][0]
                self.assertEqual(sample["text"], "@USER @USER Alte noch 4 min rot. I weiss nöd ob du die Szene gseh hesch aber da isch nöd mol annähernd e Foul gescjweige den rot und trotzdem mit 10 Mann no so guet spiele isch krass guet find ich für de fcsg")
                self.assertEqual(sample["lang"], "gsw")

            def test_num_samples(self):
                self.assertEqual(len(self.dataset.dataset["train"]), 381654)

            def test_print_statistics(self):
                self.dataset.print_statistics()


class SwissdoxDatasetTestCase(TestCase):

    def setUp(self):
        self.dataset = SwissdoxDataset()

    def test_print_statistics(self):
        self.dataset.print_statistics()

    def test_sample(self):
        sample = self.dataset.dataset['train'][0]
        print(sample)


class AdaptationDatasetTestCase(TestCase):

        def setUp(self):
            self.dataset = AdaptationDataset()

        def test_print_statistics(self):
            self.dataset.print_statistics()

        def test_sample(self):
            sample = self.dataset.dataset['train'][0]
            print(sample)


class SaveAdaptationDatasetTestCase(TestCase):

        def setUp(self):
            save_path = Path(__file__).parent.parent / 'data' / 'continued_pretraining'
            self.dataset = load_from_disk(str(save_path.resolve()))
            print(self.dataset)

        def test_sample(self):
            sample = self.dataset['train'][0]
            print(sample)
