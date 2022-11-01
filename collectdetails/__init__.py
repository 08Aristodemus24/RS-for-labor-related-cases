from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.by import By
import pandas as pd


# collects and returns links and link texts of page
def collect_link_details(driver: webdriver.Chrome, link: str, link_selector: str) -> pd.DataFrame:
    try:
        driver.get(link)
        wait_val = WebDriverWait(driver, timeout=10).until(lambda driver: driver.execute_script('return document.readyState === "complete"'))

        # narrow down using selector 
        content = driver.find_element(By.CSS_SELECTOR, link_selector)
        links = content.find_elements(By.TAG_NAME, "a")

        # collect link texts and hrefs
        link_texts = []
        link_hrefs = []
        for link in links:
            # extract href of link through get dom attribute
            # extract text of link through self.text
            link_hrefs.append(link.get_attribute('href'))
            link_texts.append(link.text)
        return pd.DataFrame({'link_text': link_texts, 'href': link_hrefs})
    except TimeoutError as error:
        print("Error {} has occured".format(error))


# collects and returns link, headers, and text content of page
def collect_content(driver: webdriver.Chrome, links: list[str], header_selector: str, text_content_selector: str) -> pd.DataFrame:
    page_headers = []
    page_text_content = []
    accepted = []
    rejects = []
    for link in links:
        # otherwise if the callback does not return a true value and time
        # period is up, then WebDriverWait raises a timeout error
        try:
            driver.get(link)
            wait_val = WebDriverWait(driver, timeout=10).until(lambda driver: driver.execute_script('return document.readyState === "complete"'))
            print("Wait value: {}\nPage title: {}\n".format(wait_val, driver.title))
            
            # grab html elements that contain important header content
            # exhaust the list returned by appending
            header = " ".join([element.text for element in driver.find_elements(By.CSS_SELECTOR, header_selector)])

            # grab a single html element only since it cannto be generalized 
            # across multiple pages since the structure may change and some
            # elements may not be grabbed
            text_content = driver.find_element(By.CSS_SELECTOR, text_content_selector)
            print("Header: {}\nText content{}\n\n".format(header, text_content.text))
            
            # store accepted headers and text content
            page_headers.append(header)
            page_text_content.append(text_content.text)

            # append if no error is raised
            accepted.append(link)
            
        except TimeoutError as error:
            print("Error {} has occured".format(error))
            driver.quit() # or .close()

        # catch both NoSuchElementException and StaleElementReferenceException errors
        except (NoSuchElementException, StaleElementReferenceException) as error:
            print("Error {} has occured".format(error))
            rejects.append(link)

        finally:
            print("will go to next link")

    print("number of total links: {}\nnumber of accepted links: {}\nnumber of reject links: {}\n".format(len(links), len(accepted), len(rejects)))
    print(accepted, end='\n')
    print(rejects, end='\n')

    # place text in data frame if no error is raised
    return pd.DataFrame({'page_link': accepted, 'page_header': page_headers, 'page_text_content': page_text_content})