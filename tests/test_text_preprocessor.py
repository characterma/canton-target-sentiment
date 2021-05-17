import unittest
import sys
sys.path.append('../src/')
from preprocess import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):

    def test_simplified_chinese(self):
        traditional = "#儀式感不能少沒有卡地亞，沒有浪琴，但是我有阿瑪尼，“我願意把星辰銀河都送給你”別說人間不值得 你最值得！"
        simplified = "#仪式感不能少没有卡地亚，没有浪琴，但是我有阿玛尼，“我愿意把星辰银河都送给你”别说人间不值得 你最值得！"
        target_locs = [[15, 17]]
        pp = TextPreprocessor(
            text=traditional, 
            target_locs=target_locs,
            steps=['simplified_chinese']
        )
        preprocessed_text = pp.preprocessed_text
        preprocessed_target_locs = pp.preprocessed_target_locs
        self.assertTrue(preprocessed_text==simplified)
        self.assertTrue(preprocessed_target_locs==target_locs)

    def test_lower_case(self):
        original = "Filled with jealousy, Omega made their thoughts known on social media."
        lower_cased = "filled with jealousy, omega made their thoughts known on social media."
        target_locs = [[23, 28]]
        pp = TextPreprocessor(
            text=original, 
            target_locs=target_locs,
            steps=['lower_case']
        )
        preprocessed_text = pp.preprocessed_text
        preprocessed_target_locs = pp.preprocessed_target_locs
        self.assertTrue(preprocessed_text==lower_cased)
        self.assertTrue(preprocessed_target_locs==target_locs)

    def test_convert_java_index(self):
        text = "#好物推薦🔥 #每日穿搭  卡地亞鑰匙系列 單表機械機芯 95❤ 滿鑽 超值💰帶走#好物推薦🔥 #每日穿搭  卡地亞鑰匙系列 單表機械機芯 95❤ 滿鑽 超值💰帶走"
        java_index = [[15, 18], [25, 27], [27, 29], [58, 61], [68, 70], [70, 72]]
        python_index = [[14, 17], [24, 26], [26, 28], [55, 58], [65, 67], [67, 69]]
        pp = TextPreprocessor(
            text=text, 
            target_locs=java_index,
            steps=['convert_java_index']
        )
        preprocessed_text = pp.preprocessed_text
        preprocessed_target_locs = pp.preprocessed_target_locs
        self.assertTrue(preprocessed_text==text)
        self.assertTrue(preprocessed_target_locs==python_index)

    def test_extract_post_context(self):
        text = "前年8‧18反修例「流水式集會」未經批准集結案中，現正在赤柱監獄還押的壹傳媒黎智英與另外8名泛民人士被檢控，案件將於明天（16日）開審，預計審期10天。其中一名被告區諾軒早前表明會認罪，法庭明天會正式聽取他的認罪答辯，而其餘8名被告均已否認控罪，當中包括黎智英。 9名被告，包括黎智英、李柱銘、何俊仁、李卓人、梁國雄、吳靄儀、梁耀忠、何秀蘭及區諾軒。他們被控一項組織未經批准集結及一項參與未經批准集結罪。律政司的檢控團隊會由資深大律師余若海領軍，而代表黎智英的是資深大律師余若薇，這場官司將由姊弟對壘。 法庭文件顯示，控方暫定有7名證人，包括6名警務人員和一名港鐵經理，並會有35段新聞報道和27段案發片段呈堂。辯方將會提出法律爭議，聲稱《公安條例》將組織或參與和平集結定為罪行，並賦權警務處處長反對集結，屬損害公民集會權利。另外，在本案所涉的集結關乎批評警隊，辯方質疑由警務處處長決定是否批准，屬有利益衝突和偏頗之嫌。此外，辯方亦會質疑公眾集會及遊行上訴委員會是否獨立公正。 OA_show(ONCC.Function.GetBanner.WriteOpenXAdZone('content_advContent',ONCC.Function.getSection(),ONCC.Function.getLocation(),ONCC.Function.getNation())); "
        target_locs = [[82, 85], [171, 174]]

        expected_text = "其中一名被告區諾軒早前表明會認罪，法庭明天會正式聽取他的認罪答辯，而其餘8名被告均已否認控罪，當中包括黎智英 9名被告，包括黎智英、李柱銘、何俊仁、李卓人、梁國雄、吳靄儀、梁耀忠、何秀蘭及區諾軒"
        expected_target_locs = [[6, 9], [94, 97]]
        pp = TextPreprocessor(
            text=text, 
            target_locs=target_locs,
            steps=['extract_post_context']
        )
        preprocessed_text = pp.preprocessed_text
        preprocessed_target_locs = pp.preprocessed_target_locs
        self.assertTrue(preprocessed_text==expected_text)
        self.assertTrue(preprocessed_target_locs==expected_target_locs)