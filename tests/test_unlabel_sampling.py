import os
import unittest
import papermill as pm
import scrapbook as sb
from pathlib import Path, PurePath

kernel_name = "python3"
notebook_path = '../notebooks/unlabel_data_sampling.ipynb'
output_notebook = "../notebooks/unlabel_data_sampling_tmp.ipynb"

class TestUnlabelSampling(unittest.TestCase):
    test_dir = Path(PurePath(__file__).parent).resolve()
    src_dir = test_dir.parent / "nlp_pipeline"
    # config_dir = test_dir.parent / "config"
    # data_dir = test_dir.parent / "data"

    @classmethod
    def setUpClass(cls):
        cls.sample_size = 10
        cls.label_ratio = {'-1': 0.3, '0': 0.2, '1': 0.5}
        cls.certainty = 0.33
        pm.execute_notebook(
            notebook_path, 
            output_notebook,
            kernel_name = kernel_name,
            parameters = dict(
                src_dir = str(cls.src_dir),
                sample_size = cls.sample_size,
                label_ratio = cls.label_ratio,
                certainty = cls.certainty
            )
        )
        cls.results = sb.read_notebook(str(cls.src_dir / output_notebook)).scraps.data_dict

    # test number of samples in result
    def test_number_of_sample(self):
        self.assertTrue(self.results['length'] == self.sample_size)

    # test number of samples in result
    def test_label_ratio(self):
        self.assertTrue(self.results['label_ratio'] == self.label_ratio)

    # test number of samples in result
    def test_certainty(self):
        self.assertTrue(self.results['min_certainty'] > self.certainty)

    @classmethod
    def tearDownClass(cls):
        os.system(f"rm {output_notebook}")