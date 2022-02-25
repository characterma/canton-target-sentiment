import unittest
import sys

from nlp_pipeline.preprocess import Preprocessor
# passed on 2022-02-16

class TestPreprocessor(unittest.TestCase):
    def test_simplified_chinese(self):
        data_dict_1 = {
            'content': "#儀式感不能少沒有卡地亞，沒有浪琴，但是我有阿瑪尼，“我願意把星辰銀河都送給你”別說人間不值得 你最值得！", 
            'target_locs': [[15, 17]]
        }

        data_dict_2 = {
            'content': "#仪式感不能少没有卡地亚，没有浪琴，但是我有阿玛尼，“我愿意把星辰银河都送给你”别说人间不值得 你最值得！", 
            'target_locs': [[15, 17]]
        }

        pp = Preprocessor(
            data_dict=data_dict_1, steps=["simplified_chinese"]
        )

        self.assertTrue(pp.data_dict['content'] == data_dict_2['content'])

    def rm_emojis(self):
        data_dict_1 = {
            'content': "#[鼓掌]仪式感不能少没有卡地亚，没有浪琴，但是我有阿玛尼，“我愿意把星辰银河都送给你”别说人间不值得 你最值得！[太开心]", 
            'target_locs': [[19, 21]]
        }

        data_dict_2 = {
            'content': "#仪式感不能少没有卡地亚，没有浪琴，但是我有阿玛尼，“我愿意把星辰银河都送给你”别说人间不值得 你最值得！", 
            'target_locs': [[15, 17]]
        }

        pp = Preprocessor(
            data_dict=data_dict_1, steps=["rm_emojis"]
        )

        self.assertTrue(pp.data_dict['content'] == data_dict_2['content'])
        self.assertTrue(pp.data_dict['target_locs'] == data_dict_2['target_locs'])


    def test_enclose_target(self):
        data_dict_1 = {
            'content': "#仪式感不能少没有卡地亚，没有浪琴，但是我有阿玛尼，“我愿意把星辰银河都送给你”别说人间不值得 你最值得！", 
            'target_locs': [[15, 17]]
        }
        data_dict_2 = {
            'content': "#仪式感不能少没有卡地亚，没有[E]浪琴[/E]，但是我有阿玛尼，“我愿意把星辰银河都送给你”别说人间不值得 你最值得！", 
            'target_locs': [[15, 18]]
        }
        pp = Preprocessor(
            data_dict=data_dict_1, steps=["enclose_target"]
        )

        self.assertTrue(pp.data_dict['content'] == data_dict_2['content'])
        self.assertTrue(pp.data_dict['target_locs'] == data_dict_2['target_locs'])

    def test_normalize_target(self):
        data_dict_1 = {
            'content': "#仪式感不能少没有卡地亚，没有浪琴，但是我有阿玛尼，“我愿意把星辰银河都送给你”别说人间不值得 你最值得！", 
            'target_locs': [[15, 17]]
        }
        data_dict_2 = {
            'content': "#仪式感不能少没有卡地亚，没有[unused1]，但是我有阿玛尼，“我愿意把星辰银河都送给你”别说人间不值得 你最值得！", 
            'target_locs': [[15, 24]]
        }
        pp = Preprocessor(
            data_dict=data_dict_1, steps=["normalize_target"]
        )

        self.assertTrue(pp.data_dict['content'] == data_dict_2['content'])
        self.assertTrue(pp.data_dict['target_locs'] == data_dict_2['target_locs'])


    def test_mask_other_targets(self):
        data_dict_1 = {
            'content': '2020年底以来,默沙东、、阿斯利康和罗氏均爆出在国外撤销PD-1L1产品适应症的消息', 
            'target_locs': [[19, 21]], 
            'other_target_locs': [[0, 6], [9, 12], [14, 18], [19, 21]], 
        }
        data_dict_2 = {
            'content': '[unused2]以来,[unused2]、、[unused2]和罗氏均爆出在国外撤销PD-1L1产品适应症的消息', 
            'target_locs': [[33, 35]]
        }
        pp = Preprocessor(
            data_dict=data_dict_1, steps=["mask_other_targets"]
        )

        self.assertTrue(pp.data_dict['content'] == data_dict_2['content'])
        self.assertTrue(pp.data_dict['target_locs'] == data_dict_2['target_locs'])


    def test_lower_case(self):
        data_dict_1 = {
            'content': "Filled with jealousy, Omega made their thoughts known on social media."
        }
        data_dict_2 = {
            'content': "filled with jealousy, omega made their thoughts known on social media."
        }
        pp = Preprocessor(
            data_dict=data_dict_1, steps=["lower_case"]
        )
        self.assertTrue(pp.data_dict['content'] == data_dict_2['content'])

    def test_full_to_half(self):
        data_dict_1 = {'content': "！？；，"}
        data_dict_2 = {'content': "!?;,"}
        pp = Preprocessor(
            data_dict=data_dict_1, steps=["full_to_half"]
        )
        self.assertEqual(pp.data_dict['content'], data_dict_2['content'])

    def test_rm_non_chinese_char(self):
        data_dict_1 = {'content': "#好rrr_物_推薦🔥 ABC"}
        data_dict_2 = {'content': "好物推薦"}
        pp = Preprocessor(
            data_dict=data_dict_1, steps=["rm_non_chinese_char"]
        )
        self.assertEqual(pp.data_dict['content'], data_dict_2['content'])
        
    def test_convert_java_index(self):
        data_dict_1 = {
            'content': "#好物推薦🔥 #每日穿搭  卡地亞鑰匙系列 單表機械機芯 95❤ 滿鑽 超值💰帶走#好物推薦🔥 #每日穿搭  卡地亞鑰匙系列 單表機械機芯 95❤ 滿鑽 超值💰帶走", 
            'target_locs': [[15, 18], [25, 27], [27, 29], [58, 61], [68, 70], [70, 72]]    
        }
        data_dict_2 = {
            'content': "#好物推薦🔥 #每日穿搭  卡地亞鑰匙系列 單表機械機芯 95❤ 滿鑽 超值💰帶走#好物推薦🔥 #每日穿搭  卡地亞鑰匙系列 單表機械機芯 95❤ 滿鑽 超值💰帶走", 
            'target_locs': [[14, 17], [24, 26], [26, 28], [55, 58], [65, 67], [67, 69]]    
        }
        pp = Preprocessor(
            data_dict=data_dict_1, steps=["convert_java_index"]
        )
        self.assertTrue(pp.data_dict["content"] == data_dict_2["content"])
        self.assertTrue(pp.data_dict["target_locs"] == data_dict_2["target_locs"])

    # def test_extract_post_context_1(self):
    #     data_dict_1 = {
    #         'content': "前年8‧18反修例「流水式集會」未經批准集結案中，現正在赤柱監獄還押的壹傳媒。黎智英與另外8名泛民人士被檢控，案件將於明天（16日）開審，預計審期10天。其中一名被告區諾軒早前表明會認罪，法庭明天會正式聽取他的認罪答辯，而其餘8名被告均已否認控罪，當中包括黎智英。 9名被告，包括黎智英、李柱銘、何俊仁、李卓人、梁國雄、吳靄儀、梁耀忠、何秀蘭及區諾軒。他們被控一項組織未經批准集結及一項參與未經批准集結罪。律政司的檢控團隊會由資深大律師余若海領軍，而代表黎智英的是資深大律師余若薇，這場官司將由姊弟對壘。 法庭文件顯示，控方暫定有7名證人。包括6名警務人員和一名港鐵經理，並會有35段新聞報道和27段案發片段呈堂。辯方將會提出法律爭議，聲稱《公安條例》將組織或參與和平集結定為罪行，並賦權警務處處長反對集結，屬損害公民集會權利。另外，在本案所涉的集結關乎批評警隊，辯方質疑由警務處處長決定是否批准，屬有利益衝突和偏頗之嫌。此外，辯方亦會質疑公眾集會及遊行上訴委員會是否獨立公正。 OA_show(ONCC.Function.GetBanner.WriteOpenXAdZone('content_advContent',ONCC.Function.getSection(),ONCC.Function.getLocation(),ONCC.Function.getNation())); ", 
    #         'target_locs': [[82, 85], [171, 174]]
    #     }
    #     data_dict_2 = {
    #         'content': "其中一名被告區諾軒早前表明會認罪，法庭明天會正式聽取他的認罪答辯，而其餘8名被告均已否認控罪，當中包括黎智英。 9名被告，包括黎智英、李柱銘、何俊仁、李卓人、梁國雄、吳靄儀、梁耀忠、何秀蘭及區諾軒", 
    #         'target_locs': [[6, 9], [95, 98]]
    #     }
    #     pp = Preprocessor(
    #         data_dict=data_dict_1, steps=["extract_post_context_1"]
    #     )

    #     self.assertTrue(pp.data_dict['content'] == data_dict_2['content'])
    #     self.assertTrue(pp.data_dict['target_locs'] == data_dict_2['target_locs'])

    # def test_extract_post_context_2(self):
    #     data_dict_1 = {
    #         'content': "前年8‧18反修例「流水式集會」未經批准集結案中，現正在赤柱監獄還押的壹傳媒。黎智英與另外8名泛民人士被檢控，案件將於明天（16日）開審，預計審期10天。其中一名被告區諾軒早前表明會認罪，法庭明天會正式聽取他的認罪答辯，而其餘8名被告均已否認控罪，當中包括黎智英。 9名被告，包括黎智英、李柱銘、何俊仁、李卓人、梁國雄、吳靄儀、梁耀忠、何秀蘭及區諾軒。他們被控一項組織未經批准集結及一項參與未經批准集結罪。律政司的檢控團隊會由資深大律師余若海領軍，而代表黎智英的是資深大律師余若薇，這場官司將由姊弟對壘。 法庭文件顯示，控方暫定有7名證人。包括6名警務人員和一名港鐵經理，並會有35段新聞報道和27段案發片段呈堂。辯方將會提出法律爭議，聲稱《公安條例》將組織或參與和平集結定為罪行，並賦權警務處處長反對集結，屬損害公民集會權利。另外，在本案所涉的集結關乎批評警隊，辯方質疑由警務處處長決定是否批准，屬有利益衝突和偏頗之嫌。此外，辯方亦會質疑公眾集會及遊行上訴委員會是否獨立公正。 OA_show(ONCC.Function.GetBanner.WriteOpenXAdZone('content_advContent',ONCC.Function.getSection(),ONCC.Function.getLocation(),ONCC.Function.getNation())); ", 
    #         'target_locs': [[82, 85], [171, 174]]
    #     }
    #     data_dict_2 = {
    #         'content': "黎智英與另外8名泛民人士被檢控，案件將於明天（16日）開審，預計審期10天。其中一名被告區諾軒早前表明會認罪，法庭明天會正式聽取他的認罪答辯，而其餘8名被告均已否認控罪，當中包括黎智英。 9名被告，包括黎智英、李柱銘、何俊仁、李卓人、梁國雄、吳靄儀、梁耀忠、何秀蘭及區諾軒。他們被控一項組織未經批准集結及一項參與未經批准集結罪", 
    #         'target_locs': [[44, 47], [133, 136]]
    #     }
    #     pp = Preprocessor(
    #         data_dict=data_dict_1, steps=["extract_post_context_2"]
    #     )
    #     self.assertTrue(pp.data_dict['content'] == data_dict_2['content'])
    #     self.assertTrue(pp.data_dict['target_locs'] == data_dict_2['target_locs'])
