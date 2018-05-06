#Selenium is a web browser testing automation tool
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import json
import os
# mv chrome driver from Downloads to Applications 
chromedriver = "/Applications/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver

driver = webdriver.Chrome(chromedriver)
driver.get("https://www.lonelyplanet.com/travel-tips-and-articles")


compose_button=driver.find_element_by_xpath('//*[@id="js-load-more"]')

for i in range(400):
	try:
		compose_button.click()
	except:
		pass



buyers = driver.find_elements_by_xpath('//div[@id="js-results"]//a')
url_list = []
for buyer in buyers:
	url_list.append(buyer.get_attribute("href"))
#print (url_list)


data = []

for i in range(len(url_list)):

	try:

		article_data = {}
		next_url = url_list[i]
		print(next_url)
		driver.get(next_url)

		theURL = driver.current_url
		print(driver.current_url)

		theText = []
		theImage = []


		theTitle = driver.find_element_by_xpath('//h1[@class="article-header__title is-xlarge"]').get_attribute("innerHTML")
		print(theTitle)
		article = driver.find_elements_by_xpath('//section[@class="page-container page-container--ready"]//div[@class="article-body__content"]//p')

		images = driver.find_elements_by_xpath('//a[@class="copy--body__link"]//img')

		for j in range(len(article)):
			theText.append(article[j].get_attribute("innerHTML"))
			#print(article[j].get_attribute("innerHTML"))

		for j in range(len(images)):
			theImage.append(images[j].get_attribute("src"))
			#print(article[j].get_attribute("innerHTML"))


		article_data = {
		'title': theTitle,
		'URL': theURL,
		'article': theText,
		'images': theImage
		}

		data.append(article_data)


	except:
		pass



with open('data_lonelyPlanet.json', 'w') as outfile:
    json.dump(data, outfile)

print(' ---------------------------- ')
print('')
print('Done')
driver.close()