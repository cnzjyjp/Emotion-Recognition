import os

#剧本角色数据集排序
def sort_data(path2: str):
    # path1 = 'data_sort.tsv'
    path1 = path2.split('.')[0] + '_sort.tsv'
    text_list = []
    if os.path.exists(path1):
        os.remove(path1)
    s = open(path1,'w',encoding='utf-8')
    with open(path2,'r',encoding='utf-8') as f:
        for l in f:
            id = l.split('\t')[0]
            # print(id)
            script_ids = id.split('_')[0]
            try:
                scene_nums = id.split('_')[1]
            except IndexError:
                print(id)
            sentence_nums = id.split('_')[3]
            text_list.append((script_ids,scene_nums,sentence_nums,l.replace('\n','')))
        text_list.sort(key=lambda x:int(x[0]))
        # print(len(text_list)) # 42790
        # raise Exception("TEST")
        n1 = 0
        while n1<len(text_list):
            scene_list = [(i[1],i[2],i[3]) for i in text_list if text_list[n1][0]==i[0]]
            n1 += len(scene_list)
            scene_list.sort(key=lambda x:int(x[0]))
            # print(n1)
            # print(scene_list[0], scene_list[1])
            # raise Exception("TEST")
            n2 = 0
            while n2<len(scene_list):
                sentence_list = [(i[1], i[2]) for i in scene_list if scene_list[n2][0] == i[0]]
                n2 += len(sentence_list)
                sentence_list.sort(key=lambda x:int(x[0]))
                for t in sentence_list:
                    s.write(t[1]+'\n')


    f.close()
    s.close()

sort_data('train.tsv')
sort_data('test.tsv')
maxlen = 350
#训练集生成
if os.path.exists('train_final.tsv'):
    os.remove('train_final.tsv')
s = open('train_final.tsv','w',encoding='utf-8')
data = {}
target = ''
with open('train_sort.tsv','r',encoding='utf-8') as f:
    for l in f.readlines():
        if target!=l.split('\t')[0].split('_')[0]:
            data = {}
        character = l.split('\t')[2].replace('\n', '')
        content = l.split('\t')[1]
        label = l.split('\t')[3]
        id = l.split('\t')[0]
        if character!='':
            if data.get(character)!=None:
                text = data[character] + '（' + '角色是:' + character + '）'+ content
                up_text = data.get(character) + content
                data[character] = up_text
            else:
                text = '（' + '角色是:' + character + '）' + content
                data[character] = content
            if len(text) > maxlen:
                old_text = ''
                text_list =  [i for i in text.split('。') if i!='']
                for t in range(len(text_list)):
                    if len(text_list[len(text_list) - 1 - t] + '。' + old_text) < maxlen:
                        old_text = text_list[len(text_list) - 1 - t] + '。' + old_text
                    else:
                        break
                if old_text != '':
                    text = old_text
                else:
                    text = text[(len(text)-1-maxlen+len(content)):]

        else:text = '（' + '角色是:' + character + '）' + content
        if text == '':
            text = '（' + '角色是:' + character + '）' + content
        if label!='' and len(label.split(','))==6:
            s.write(id+'\t'+text+'\t'+label)
        target = l.split('\t')[0].split('_')[0]

f.close()
s.close()

#测试集生成
if os.path.exists('test_final.tsv'):
    os.remove('test_final.tsv')
s = open('test_final.tsv','w',encoding='utf-8')
data = {}
target = ''
with open('test_sort.tsv','r',encoding='utf-8') as f:
    for l in f.readlines():
        if target!=l.split('\t')[0].split('_')[0]:
            data = {}
        character = l.split('\t')[2].replace('\n', '')
        content = l.split('\t')[1]
        id = l.split('\t')[0]
        if character!='':
            if data.get(character)!=None:
                text = data[character] + '（' + '角色是:' + character + '）'+ content
                up_text = data.get(character) + content
                data[character] = up_text
            else:
                text ='（' + '角色是:' + character + '）' + content
                data[character] = content
            if len(text) > maxlen:
                old_text = ''
                text_list = [i for i in text.split('。') if i!='']
                for t in range(len(text_list)):
                    if len(text_list[len(text_list) - 1 - t] + '。' + old_text) < maxlen:
                        old_text = text_list[len(text_list) - 1 - t] + '。' + old_text
                    else:
                        break
                if old_text != '' :
                    text = old_text
                else:
                    text = text[(len(text) - 1 - maxlen + len(content)):]
        else:text = '（' + '角色是:' + character + '）'+ content
        if text == '':
            text = '（' + '角色是:' + character + '）' + content
        s.write(id + '\t' + text + '\n')
        target = l.split('\t')[0].split('_')[0]

f.close()
s.close()
