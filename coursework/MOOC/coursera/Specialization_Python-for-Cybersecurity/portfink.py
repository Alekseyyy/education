
# PORTFINK: Now you're gonna do the hacking ;-)
# By Aleksey
# - https://github.com/Alekseyyy
# - https://keybase.io/epsiloncalculus

# Here if you don't get the reference: https://youtu.be/jn_dKqtAUjE

import streamlit as st

from selenium import webdriver
from selenium.webdriver.firefox.options import Options

class osint:

	def __init__(self):
		self.options = Options()
		self.options.headless = True
		self.driver = webdriver.Firefox(options=self.options, executable_path="./geckodriver")

	def search_name(self, first, last):
		self.driver.get("https://duckduckgo.com/?q=%s+%s" % (first, last))
		title = self.driver.find_elements_by_class_name("js-result-title-link")
		url = [k.get_attribute("href") for k in title]
		st.markdown("### Name search query results")
		st.markdown("<br/>".join(["<a href=\"%s\">%s</a>" % (y, x.text) for x, y in zip(title, url)]), unsafe_allow_html = True)

def main():
	st.markdown("""
		<div>
		<h1>PORTFINK</h1>
		<p>Now you're gonna do the hacking! <br/> https://github.com/Alekseyyy</p>
		</p>
		</div>
	""", unsafe_allow_html = True) #til that for streamlit's markdown, you can add HTML by setting its "unsafe_allow_html" parameter to True!

	st.sidebar.markdown("### Target information")
	param1 = st.sidebar.text_input("First name")
	param2 = st.sidebar.text_input("Last name")
	param3 = st.sidebar.text_input("Company")
	param4 = st.sidebar.text_input("Address")
	param5 = st.sidebar.text_input("Phone")
	param6 = st.sidebar.text_input("IP to scan")

	# Get scanning
	if st.sidebar.button("Scan"):
		st.markdown("## Results")
		task_osint = osint()
		task_osint.search_name(param1, param2)

if __name__ == "__main__":
	main()

