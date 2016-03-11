#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup

r = requests.get('http://www.thesaurus.com/browse/anger')
headingRe = re.compile('<div class="heading-row">.*<section class="container-info antonyms">', re.DOTALL)
wordRe = re.compile('<span class="text">([^\/].*)</span>')

synonyms = soup.find_all('div', class_='heading-row')[0]

