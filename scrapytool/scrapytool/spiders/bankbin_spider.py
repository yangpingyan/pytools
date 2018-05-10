# -*- coding: utf-8 -*-

import scrapy
from scrapy.spider import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.item import Item, Field


class MyItem(scrapy.Item):
    value = scrapy.Field()

class BankbinSpider(scrapy.Spider):
    name = "bankbin"
    allowd_domains = ['http://www.5cm.cn/']
    start_urls = ['http://www.5cm.cn/bank/',]
    
    
    def parse(self, response):
        url_suffix = ['nanjing/', 'suzhou/', 'nantong/', 'wuxi/', 'changzhou/', 'yangzhou/', 'yancheng/',
                      'xuzhou/', 'taizhou/', 'zhenjiang/', 'lianyungang/', 'huaian/', 'suqian/',
                      'hangzhou/', 'ningbo/', 'wenzhou/', 'jinhua/', 'shaoxing/', 'taizhou1/', 
                      'jiaxing/', 'huzhou/', 'lishui/', 'zhoushan/', 'quzhou/',]
        for href in response.xpath('//ul[@class="list-unstyled"]/li/a/@href'):
            url_main = response.urljoin(href.extract())
            
#            print("url {}".format(url_main))
            for url_s in url_suffix:
                full_url = url_main + url_s
#                print("url {}".format(full_url))
                yield scrapy.Request(full_url, callback=self.parse_item)
                
    def parse_item(self, response):        
        for quote in response.xpath('//table[@class="table"]/tr'):
            yield {
                'hanghao': quote.xpath('.//td[1]/text()').extract_first(),
                'fenhan': quote.xpath('.//td/a/text()').extract_first(),
#                'dianhua': quote.xpath('.//td[3]/text()').extract_first(),
#                'youbian': quote.xpath('.//td[4]/text()').extract_first(),
#                'dizhi': quote.xpath('.//td[5]/text()').extract_first(),
#                'swift code': quote.xpath('.//td[6]/text()').extract_first(),
               
            }
                
        next_page = response.xpath('//li/a[@class="next"]/@href').extract_first()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse_item)


if __name__ == '__main__':
    print('MISSION START')
