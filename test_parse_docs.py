from parse_docs import PLACEHOLDER_SKELETONS, match_placeholder

# match_placeholder
cases = []
for key, skeleton in PLACEHOLDER_SKELETONS.items():
    cases.append((skeleton, key))
cases.append(('（二）异地就医办理地址：XX市XX区（市、县）XX路XX号XX服务大厅XX-XX号窗口', 'address'))
cases.append((PLACEHOLDER_SKELETONS['publish_date'].replace('                   ', ' '), 'publish_date'))
for text in ['南宁XX贸易公司', 'zwfw.gxzf.gov.cn', '胸部X线检查报告', '', '地址：南宁市青秀区民族大道100号服务大厅1-1号窗口']:
    cases.append((text, None))

n_ok = 0
n_fail = 0
for text, want in cases:
    got = match_placeholder(text)
    if got == want:
        n_ok = n_ok + 1
        print('OK ', repr(text[:30]), '->', got, '期望', want)
    else:
        n_fail = n_fail + 1
        print('FAIL', repr(text[:30]), '->', got, '期望', want)
print('合计', len(cases), '条：OK', n_ok, '/ FAIL', n_fail)