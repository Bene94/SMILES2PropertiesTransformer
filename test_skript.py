

#save google.com as png
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

driver = webdriver.Chrome()
driver.get("https://www.google.com")
time.sleep(2)

driver.save_screenshot("google.png")
