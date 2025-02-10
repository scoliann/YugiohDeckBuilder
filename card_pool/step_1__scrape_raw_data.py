

# Do imports
import os
import time
import pandas as pd
import collections as cl
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm


# Do local imports
from args import s_card_pool_url
from yugioh_metadata import monster_attributes


# Capitalize monster attributes
monster_attributes = [s_attribute.upper() for s_attribute in monster_attributes]


def main():

    # Get directory name
    s_format_dir = s_card_pool_url.split('format=')[-1]

    # Create format directory
    os.makedirs(s_format_dir, exist_ok=True)

    # Define output file
    s_pickle_file = os.path.join(s_format_dir, 'card_pool_raw.pkl')
    
    # Initialize webdriver
    o_options = Options()
    o_options.add_argument('--start-maximized')
    o_driver = webdriver.Chrome(options=o_options)

    # Open page in selenium
    o_driver.get(s_card_pool_url)
    print('loading page...')

    # Scroll down
    o_driver.execute_script('window.scrollBy(0, 400);')
    print('scrolling...')

    # Locate the dropdown element
    o_dropdown = o_driver.find_element(By.ID, 'cardsPerPageSelector')
    o_dropdown.click()
    print('dropping down...')

    # Make selection from dropdown
    o_option = o_driver.find_element(By.XPATH, '//option[text()="100 Cards / Page"]')
    o_option.click()
    time.sleep(10)
    print('clicking selection...')

    # Page through card pool
    d_data = cl.defaultdict(list)
    while True:

        # Get card elements
        lo_cards = o_driver.find_elements(By.XPATH, '//*[@class="odd-search-results-row" or @class="even-search-results-row"]')
        print('getting cards...')

        # Iterate over cards
        for o_card in tqdm(lo_cards, desc='Scraping Page'):

            # Extract data
            s_name = o_card.find_element(By.CLASS_NAME, 'inner-cardRow-table').find_elements(By.TAG_NAME, 'tr')[0].find_elements(By.TAG_NAME, 'th')[0].text.strip()
            s_tcg_release = o_card.find_element(By.CLASS_NAME, 'inner-cardRow-table').find_elements(By.TAG_NAME, 'tr')[0].find_elements(By.TAG_NAME, 'th')[1].text.strip()
            s_attribute = o_card.find_element(By.CLASS_NAME, 'inner-cardRow-table').find_elements(By.TAG_NAME, 'tr')[1].find_elements(By.TAG_NAME, 'td')[0].text.strip()
            s_category = 'MONSTER' if s_attribute in monster_attributes else s_attribute
            if s_category == 'MONSTER':
                s_level = o_card.find_element(By.CLASS_NAME, 'inner-cardRow-table').find_elements(By.TAG_NAME, 'tr')[1].find_elements(By.TAG_NAME, 'td')[1].text.strip()
                s_type = o_card.find_element(By.CLASS_NAME, 'inner-cardRow-table').find_elements(By.TAG_NAME, 'tr')[1].find_elements(By.TAG_NAME, 'td')[2].text.strip()
                s_atk = o_card.find_element(By.CLASS_NAME, 'inner-cardRow-table').find_elements(By.TAG_NAME, 'tr')[1].find_elements(By.TAG_NAME, 'td')[3].text.strip()
                s_def = o_card.find_element(By.CLASS_NAME, 'inner-cardRow-table').find_elements(By.TAG_NAME, 'tr')[1].find_elements(By.TAG_NAME, 'td')[4].text.strip()
                s_desc = o_card.find_element(By.CLASS_NAME, 'inner-cardRow-table').find_element(By.CLASS_NAME, 'cardrow-description').text.strip()
            elif s_category == 'SPELL':
                s_level = 'N/A'
                s_type = o_card.find_element(By.CLASS_NAME, 'inner-cardRow-table').find_elements(By.TAG_NAME, 'tr')[1].find_elements(By.TAG_NAME, 'td')[1].text.strip()
                s_atk = 'N/A'
                s_def = 'N/A'
                s_desc = o_card.find_element(By.CLASS_NAME, 'inner-cardRow-table').find_element(By.CLASS_NAME, 'cardrow-description').text.strip()
            elif s_category == 'TRAP':
                s_level = 'N/A'
                s_type = o_card.find_element(By.CLASS_NAME, 'inner-cardRow-table').find_elements(By.TAG_NAME, 'tr')[1].find_elements(By.TAG_NAME, 'td')[1].text.strip()
                s_atk = 'N/A'
                s_def = 'N/A'            
                s_desc = o_card.find_element(By.CLASS_NAME, 'inner-cardRow-table').find_element(By.CLASS_NAME, 'cardrow-description').text.strip()
            s_img_url = o_card.find_element(By.CLASS_NAME, 'card-image').get_attribute('src')

            # Store
            d_data['tcg_release'].append(s_tcg_release)
            d_data['name'].append(s_name)
            d_data['category'].append(s_category)
            d_data['attribute'].append(s_attribute)
            d_data['level'].append(s_level)
            d_data['type'].append(s_type)
            d_data['atk'].append(s_atk)
            d_data['def'].append(s_def)
            d_data['desc'].append(s_desc)
            d_data['img_url'].append(s_img_url)

        # Break if we're on the last page
        i_cards_showing = int(o_driver.find_element(By.CLASS_NAME, 'results').text.strip().split(' - ')[-1].split(' of')[0])
        i_cards_total = int(o_driver.find_element(By.CLASS_NAME, 'results').text.strip().split('of ')[-1])
        print(f'\nScraped {i_cards_showing} of {i_cards_total}')
        if i_cards_showing == i_cards_total:
            break

        # Next page
        o_page_btns = o_driver.find_element(By.CLASS_NAME, 'pagination').find_elements(By.CLASS_NAME, 'page-button')
        if o_page_btns[-1].text == '':
            o_next = o_page_btns[-2]
        else:
            o_next = o_page_btns[-1]
        o_next.click()
        time.sleep(10)

    # Structure as dataframe
    df = pd.DataFrame(d_data)

    # Close webdriver
    o_driver.close()

    # Save
    df.to_pickle(s_pickle_file)


if __name__ == '__main__':
    main()
