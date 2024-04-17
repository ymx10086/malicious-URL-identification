#Copyright 2023(dtw)
import multiprocessing
import os

import pandas
import requests
import csv
import openpyxl
import concurrent.futures as cf
import re
import urllib.parse
import urllib.request
import os
import zipfile
from alexa import Alexa
from selenium import webdriver

#所有数据都在head_info里面，结构是两层字典{"https://baidu.com":{"url":"https://baidu.com","visit":1,以及headers的信息}}
head_info={}
headers = {
    "referer": "https://www.baidu.com.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36"
}
ranker = Alexa()  # this might take a while as the rankings are downloaded

#并行函数
def parallel_do(func, args_list, max_workers=None, mode='thread'):
    max_workers = 4 if not max_workers else max_workers
    exe = cf.ThreadPoolExecutor(max_workers=4) if mode == 'thread' else cf.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
    with exe as executor:
        executor.map(func, args_list)
        executor.shutdown(wait=True)

#爬取单个url
def parallel_crawler(url2):
    head_info[url2] = {"url": url2,"rank":ranker.getrank(url2)}
    try:
        res = requests.get(url2, headers=headers, timeout=20)  # 伪装请求头，request.post()
        head_info[url2].update(res.headers)#字典合并，将{"url": url2}和res.headers合并
        head_info[url2]["visit"] = 1

        # driver = webdriver.Chrome()
        # driver.get(url2)
        # # 截图，图片后缀最好为.png，如果是其他的执行的时候会有警告，但不会报错
        # driver.get_screenshot_as_file(r"D:/picture/{}.png".format(url2[url2.find("//")+2:-1]))
        # driver.quit()

    except Exception as ex:
        print(ex)
        head_info[url2]["visit"] = 0
    else:
        print("Done!")
    #print(url2,head_info[url2])

#处理单个文件
def crawler(path):
    web = []
    if path[-1]=='v':
        with open(path, 'r', encoding='utf-8') as file_obj:
            # 1.创建reader对象
            reader = csv.reader(file_obj)
            # 2.遍历进行读取数据
            for r in reader:
                web.append(r[0])
    else:
        wb = openpyxl.load_workbook(path)  # 读取文件路径
        # 打开指定的工作簿中的指定工作表：
        ws = wb["Sheet1"]
        ws = wb.active  # 打开激活的工作表
        ws = list(ws.values)  # 转为列表
        # 2.遍历进行读取数据
        for r in ws:
            web.append(r[0])

    http_https_web=[]
    for url in web:
        if "http" in url:
            http_https_web.append(url)
            continue
        else:
            http_https_web.append("http://"+url)#两种都加，防止一种失效
            http_https_web.append("https://"+url)
    #print(http_https_web)
    try:
        parallel_do(parallel_crawler,http_https_web)#并行访问文件中所有url
    except Exception as ex:
        print(ex)
    else:
        print("Done!")
    return head_info

class Alexa:
    '''
    this class provides access to the Alexa ranking of URLs
    usage: create a new instance of this class (ranker = Alexa()) and use the getrank method
    '''
    __domain_list = []

    def __init__(self):
        try:
            # download the file
            f = open('top-1m.zip', 'wb')
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            file = opener.open('http://s3.amazonaws.com/alexa-static/top-1m.csv.zip').read()
            f.write(file)
            f.close()
            # unzip it
            current_dir = os.getcwd()
            zip = zipfile.ZipFile(r'top-1m.zip')
            zip.extractall(current_dir)
            # read the alexa ranking
            f_csv = open('top-1m.csv')
            csv_data = f_csv.read()
            f_csv.close()
            lines = csv_data.split("\n")
            for line in lines:
                try:
                    url = line.split(",")[1]
                    url = re.sub('^www\.', '', url)
                    self.__domain_list.append(url)
                except:
                    continue
        except:
            raise

    def getrank(self, url):
        ''' getrank returns the alexa rank of the domain of the given URL, or -1 if it is over 1M'''
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme == '':
            return self.getrank('http://' + url)
        domain = parsed_url.netloc
        domain = re.sub('^www\.', '', domain)
        if domain in self.__domain_list:
            return self.__domain_list.index(domain) + 1
        return -1

if __name__ == "__main__":
    for root,dirs,files in os.walk(r"./train"):#获取train中所有文件
        for file in files:
            # if file!="train1补充.xlsx":
            #     continue
            head_info=crawler(os.path.join(root,file))

            print(head_info)

            with open('test.txt', 'w') as f:
                f.write(str(head_info))
            df = pandas.DataFrame(head_info).T
            df.to_csv("finish_{}.csv".format(file))
