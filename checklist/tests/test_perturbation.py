import re
import sys
import unittest

sys.path.append("../sentiment/")
from nlp_pipeline.utils import set_seed
from perturbation import Perturbation


class TestPerturbationOnDoc(unittest.TestCase):

    def setUp(self):
        self.data = [{"content": "#苹果发布会# 说实话入耳的airpods pro 我带得不是很舒服 [泪]但是是大侠送的"}]
        self.company = ["苹果", "谷歌"]
        self.keywords = ["airpods pro", "airpods 3"]
        self.dict = {"说实话": ["说真的"]}
        self.ner_endpoint = "http://ess25.wisers.com/playground/ner-kd-jigou-gpu/entityextr/analyse"

    def test_remove_random(self):
        results_text, _ = Perturbation.perturb(
            self.data, Perturbation.remove_keyword,
            keywords=self.keywords,
            n_samples=1,
        )
        orig_text = results_text[0][0]
        auged_text = results_text[0][1]
        self.assertEqual(orig_text, self.data[0]["content"])
        self.assertNotEqual(orig_text, auged_text)
        self.assertTrue("不是" in auged_text)

    def test_remove_keyword(self):
        results_text, _ = Perturbation.perturb(
            self.data, Perturbation.remove_keyword,
            keywords=self.keywords,
            n_samples=1
        )
        orig_text = results_text[0][0]
        auged_text = results_text[0][1]
        self.assertEqual(orig_text, self.data[0]["content"])
        self.assertNotEqual(orig_text, auged_text)
        self.assertEqual(auged_text, re.sub("airpods pro", "", self.data[0]["content"]))

    def test_remove_entity(self):
        results_text, _ = Perturbation.perturb(
            self.data, Perturbation.remove_entity,
            api_url=self.ner_endpoint,
            target_type="company",
            n_samples=1
        )
        orig_text = results_text[0][0]
        auged_text = results_text[0][1]
        self.assertEqual(orig_text, self.data[0]["content"])
        self.assertNotEqual(orig_text, auged_text)
        self.assertEqual(auged_text, re.sub("苹果", "", self.data[0]["content"]))

    def test_change_entity(self):
        results_text, _ = Perturbation.perturb(
            self.data, Perturbation.change_entity,
            api_url=self.ner_endpoint,
            alternatives=self.company,
            target_type="company",
            n_samples=1
        )
        orig_text = results_text[0][0]
        auged_text = results_text[0][1]
        self.assertEqual(orig_text, self.data[0]["content"])
        self.assertNotEqual(orig_text, auged_text)
        self.assertEqual(auged_text, re.sub("苹果", "谷歌", self.data[0]["content"]))

    def test_change_keyword(self):
        results_text, _ = Perturbation.perturb(
            self.data, Perturbation.change_keyword,
            keywords=self.keywords,
            n_samples=1
        )
        orig_text = results_text[0][0]
        auged_text = results_text[0][1]
        self.assertEqual(orig_text, self.data[0]["content"])
        self.assertNotEqual(orig_text, auged_text)
        self.assertEqual(auged_text, re.sub("airpods pro", "airpods 3", self.data[0]["content"]))

    def test_change_dict(self):
        results_text, _ = Perturbation.perturb(
            self.data, Perturbation.change_dict,
            replacement_dict=self.dict,
            n_samples=1
        )
        orig_text = results_text[0][0]
        auged_text = results_text[0][1]
        self.assertEqual(orig_text, self.data[0]["content"])
        self.assertNotEqual(orig_text, auged_text)
        self.assertEqual(auged_text, re.sub("说实话", "说真的", self.data[0]["content"]))


class TestPerturbationOnSubj(unittest.TestCase):

    def setUp(self):
        self.data = [
            {'docid': '20220124A06Q2PC',
             'content': '粵財控股115億元拿下南粵銀行近六成股份,"入主"資格已獲監管批准',
             'text_subjs': {'id': '1',
                            'name': '南粵銀行',
                            'kw_idxs': [[[11, 15]]],
                            'text_idxs': [[0, 33]]},
             'label': -1}]
        self.company = ["苹果"]
        self.keywords = ["粵財控股", "南粵銀行"]
        self.dict = {"入主": ["入驻"]}
        self.ner_endpoint = "http://ess25.wisers.com/playground/ner-kd-jigou-gpu/entityextr/analyse"

    def test_remove_random(self):
        results_text, results_dict = Perturbation.perturb(
            self.data, Perturbation.remove_keyword,
            keywords=self.keywords,
            n_samples=1,
        )
        orig_text = results_text[0][0]
        auged_text = results_text[0][1]
        self.assertEqual(orig_text, self.data[0]["content"])
        self.assertNotEqual(orig_text, auged_text)
        self.assertTrue("南粵銀行" in auged_text)

    def test_remove_keyword(self):
        _, results_dict = Perturbation.perturb(
            self.data, Perturbation.remove_keyword,
            keywords=self.keywords,
            n_samples=1
        )

        orig = results_dict[0][0]
        auged = results_dict[0][1]

        orig_text = orig["content"]
        auged_text = auged["content"]

        orig_kw = orig["text_subjs"]["name"]
        auged_kw = auged["text_subjs"]["name"]

        auged_kwidx = auged["text_subjs"]["kw_idxs"][0][0]

        self.assertEqual(orig_text, self.data[0]["content"])
        self.assertNotEqual(orig_text, auged_text)
        self.assertEqual(auged_text, re.sub("粵財控股", "", self.data[0]["content"]))
        self.assertEqual(orig_kw, auged_kw)
        self.assertEquals(auged_kwidx, [7, 11])

    def test_remove_entity(self):
        _, results_dict = Perturbation.perturb(
            self.data, Perturbation.remove_entity,
            api_url=self.ner_endpoint,
            target_type="company",
            n_samples=1
        )
        orig = results_dict[0][0]
        auged = results_dict[0][1]

        orig_text = orig["content"]
        auged_text = auged["content"]

        orig_kw = orig["text_subjs"]["name"]
        auged_kw = auged["text_subjs"]["name"]

        auged_kwidx = auged["text_subjs"]["kw_idxs"][0][0]

        self.assertEqual(orig_text, self.data[0]["content"])
        self.assertNotEqual(orig_text, auged_text)
        self.assertEqual(auged_text, re.sub("粵財控股", "", self.data[0]["content"]))
        self.assertEqual(orig_kw, auged_kw)
        self.assertEquals(auged_kwidx, [7, 11])

    def test_change_entity(self):
        _, results_dict = Perturbation.perturb(
            self.data, Perturbation.change_entity,
            api_url=self.ner_endpoint,
            alternatives=self.company,
            target_type="company",
            n_samples=1
        )
        orig = results_dict[0][0]
        auged = results_dict[0][1]

        orig_text = orig["content"]
        auged_text = auged["content"]

        orig_kw = orig["text_subjs"]["name"]
        auged_kw = auged["text_subjs"]["name"]

        auged_kwidx = auged["text_subjs"]["kw_idxs"][0][0]

        self.assertEqual(orig_text, self.data[0]["content"])
        self.assertNotEqual(orig_text, auged_text)

        self.assertEqual(auged_text, re.sub("南粵銀行", "苹果", self.data[0]["content"]))
        self.assertEqual(auged_kw, "苹果")
        self.assertEquals(auged_kwidx, [11, 13])

    def test_change_keyword(self):
        _, results_dict = Perturbation.perturb(
            self.data, Perturbation.change_keyword,
            keywords=self.keywords,
            n_samples=1
        )

        orig = results_dict[0][0]
        auged = results_dict[0][1]

        orig_text = orig["content"]
        auged_text = auged["content"]

        orig_kw = orig["text_subjs"]["name"]
        auged_kw = auged["text_subjs"]["name"]

        auged_kwidx = auged["text_subjs"]["kw_idxs"][0][0]

        self.assertEqual(orig_text, self.data[0]["content"])
        self.assertNotEqual(orig_text, auged_text)

        self.assertEqual(auged_text, re.sub("南粵銀行", "粵財控股", self.data[0]["content"]))
        self.assertEqual(auged_kw, "粵財控股")
        self.assertEquals(auged_kwidx, [11, 15])

    def test_change_dict(self):
        _, results_dict = Perturbation.perturb(
            self.data, Perturbation.change_dict,
            replacement_dict=self.dict,
            n_samples=1
        )

        orig = results_dict[0][0]
        auged = results_dict[0][1]

        orig_text = orig["content"]
        auged_text = auged["content"]

        orig_kw = orig["text_subjs"]["name"]
        auged_kw = auged["text_subjs"]["name"]

        auged_kwidx = auged["text_subjs"]["kw_idxs"][0][0]

        self.assertEqual(orig_text, self.data[0]["content"])
        self.assertNotEqual(orig_text, auged_text)

        self.assertEqual(auged_text, re.sub("入主", "入驻", self.data[0]["content"]))
        self.assertEqual(auged_kw, orig_kw)
        self.assertEquals(auged_kwidx, [11, 15])


if __name__ == '__main__':
    set_seed(42)
    unittest.main()
