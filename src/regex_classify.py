import re

_target_body = r'[a-zA-Z0-9\u4E00-\u9FA5\-_]{1,}'
_regex_rules = {
    'neutral': [
        fr'(#{_target_body})', 
        fr'(@{_target_body})', 
        fr'(•{_target_body})', 
    ],
    'positive': [
        fr'({_target_body}很好用)'
    ],
    'negative': []
}

for senti, rules in _regex_rules.items():
    _regex_rules[senti] = [re.compile(r) for r in rules]

def regex_classify(text, st_idx, end_idx):
    results = []
    for senti, rules in _regex_rules.items():
        for r in rules:
            for m in r.finditer(text):
                span = m.span()
                if span[0] >= st_idx and end_idx <= span[1]:
                    return senti 
    return None 
    