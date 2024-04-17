import re
import urllib
from dateutil.parser import parse as date_parse
from urllib.parse import urlparse
import numpy as np
import math
import os
import csv
import pandas as pd
class FeatureExtraction:
    features=[]
    def __init__(self,url):
        self.features=[]
        self.url=url
        self.parturl=""
        self.wash_url()
        self.url_trans_token()

        # 判断字符x是否为字母
    def is_letter(self,x):
        if (97 <= ord(x) <= 122) or (65 <= ord(x) <= 90):
            return True
        return False

        # 判断字符x是否为数字
    def is_digit(self,x):
        if 48 <= ord(x) <= 57:
            return True
        return False
    # 计算熵
    def calculate_entropy(self,string):
        str_list = list(string)
        n = len(str_list)
        str_list_single = list(set(str_list))
        num_list = []
        for i in str_list_single:
            num_list.append(str_list.count(i))
        entropy = 0
        for j in range(len(num_list)):
            entropy += -1 * (float(num_list[j] / n)) * math.log(float(num_list[j] / n), 2)
        if len(str(entropy).split('.')[-1]) >= 7:
            return ('%.7f' % entropy)
        else:
            return (entropy)

    #计算方差
    def getLength_std(self,str_list):
        if len(str_list) == 0:
            return (0, 0)
        str_list_len = 0
        str_list_len_count = []
        str_list_std = 0
        for str in str_list:
            length = len(str)
            str_list_len += length
            str_list_len_count.append(length)
        str_arr = np.array(str_list_len_count)
        str_list_std = str_arr.std()
        return (str_list_len, str_list_std)
    #是否包含post    
    def Usinghost(self):
        try:
            port=self.url.split(":")
            if len(port)>1:
                return 1
            return -1
        except:
            return -1
    def wash_url(self):
        decode_url=urllib.parse.unquote(self.url)
        url=decode_url

        http_p = url.find('http://')
        if http_p != -1 and http_p < url.find('/'):
            url = url[http_p + 7:]

        https_p = url.find('https://')
        if https_p != -1 and https_p < url.find('/'):
            url = url[https_p + 8:]

        www_p = url.find('www')
        if www_p != -1 and www_p < url.find('.'):
            url = url[www_p + 4:]

        other_p = url.find('://')
        if other_p != -1 and other_p < url.find('/'):
            url = url[other_p + 4:]
        

        host_tail = url.find('/')
        if host_tail != -1:
            url_part = url[:host_tail]
        else:
            url_part = url
        self.parturl=url_part      
    def url_trans_token(self):
        url_part=self.parturl
        hostname_ch_n = 0
        hostname_letter_num = 0
        hostname_dig_num = 0
        hostname_point_n = 0

        for i in range(len(url_part)):
            if self.is_letter(url_part[i]):
                hostname_letter_num += 1
            elif self.is_digit(url_part[i]):
                hostname_dig_num += 1
            elif url_part[i] == '.':
                hostname_point_n += 1
            else:
                hostname_ch_n += 1

        hostname_dig_ratio = hostname_dig_num / len(url_part)
        hostname_letter_ratio = hostname_letter_num / len(url_part)

        hostname = re.split('[-_/&.()<>^@!#$*=+~:; ]', url_part)
        hostname_entropy = self.calculate_entropy(url_part)
        hostname_len, hostname_std = self.getLength_std(hostname)
        # 点的个数,特殊字符个数,数字比例,字母比例,token个数,token总长度,方差，熵,host，IP
        feature1=[hostname_point_n,hostname_ch_n,hostname_dig_ratio,hostname_letter_ratio,len(hostname),hostname_len,hostname_std,hostname_entropy,self.Usinghost()]
        self.features.extend(feature1)

def reader(path):
    web_feature=[]
    with open(path, 'r', encoding='utf-8') as file_obj:
        # 1.创建reader对象
        reader = csv.reader(file_obj)
        # 2.遍历进行读取数据
        for r in reader:
            web_feature.append(FeatureExtraction(r[0]).features)
    ff=pd.DataFrame(web_feature,index=None,columns=['hostname_point_n','hostname_ch_n','hostname_dig_ratio','hostname_letter_ratio','len_hostname','hostname_len','hostname_std','hostname_entropy','Usinghost'])
    ff.to_csv("1.csv")

if __name__ == "__main__":
    for root,dirs,files in os.walk(r"./train"):#获取train中所有文件
        for file in files:
            reader(os.path.join(root,file))


