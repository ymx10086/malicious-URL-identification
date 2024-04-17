#Copyright 2023(dtw)
import multiprocessing
import os
import requests
import csv
import openpyxl
import concurrent.futures as cf

#所有数据都在head_info里面，结构是两层字典{"https://baidu.com":{"url":"https://baidu.com","visit":1,以及headers的信息}}
head_info={}
headers = {
    "referer": "https://www.baidu.com.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36"
}

#并行函数
def parallel_do(func, args_list, max_workers=None, mode='thread'):
    max_workers = 4 if not max_workers else max_workers
    exe = cf.ThreadPoolExecutor(max_workers=4) if mode == 'thread' else cf.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
    with exe as executor:
        executor.map(func, args_list)
        executor.shutdown(wait=True)

#爬取单个url
def parallel_crawler(url2):
    head_info[url2] = {"url": url2}
    try:
        res = requests.get(url2, headers=headers, timeout=20)  # 伪装请求头，request.post()
        head_info[url2].update(res.headers)#字典合并，将{"url": url2}和res.headers合并
        head_info[url2]["visit"] = 1
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

if __name__ == "__main__":
    for root,dirs,files in os.walk(r"./train"):#获取train中所有文件
        for file in files:
            head_info=crawler(os.path.join(root,file))