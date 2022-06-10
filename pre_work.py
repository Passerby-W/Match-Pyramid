import json

# json文件保存后，汉字显示为unicode编码：
# {"": 0, "\u7814": 1, "\u7a76": 2, "\u53d1": 3, "\u73b0": 4, "\u7ec6": 5, "\u80de": 6}
# {"": 0, "研": 1, "究": 2, "发": 3, "现": 4, "细": 5, "胞": 6}
# 解决方法：

# with open('data.json', 'r', encoding='utf8') as f:
#     data = json.load(f)
#
# with open('data_cn.json', 'w', encoding='utf8') as f:
#     json.dump(data, f, ensure_ascii=False)




