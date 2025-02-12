# Do imports
import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


# Do local imports
from args import s_format


def main():

    # Get webpage
    s_ban_list_url = 'https://www.formatlibrary.com/formats/' + s_format

    # Create format directory
    os.makedirs(s_format, exist_ok=True)

    # Define output file
    s_ban_list_file = os.path.join(s_format, 'ban_list.csv')
    
    # Initialize webdriver
    o_options = Options()
    o_options.add_argument('--start-maximized')
    o_driver = webdriver.Chrome(options=o_options)

    # Open page in selenium
    o_driver.get(s_ban_list_url)
    time.sleep(5)
    print('loading page...')

    # Get forbidden cards
    try:
        ls_forbidden_cards = [
            o_elem.get_attribute('alt') for o_elem in o_driver.find_element(By.CLASS_NAME, 'banlist').find_element(By.ID, 'forbidden').find_elements(By.CLASS_NAME, 'CardImages')
        ]
    except:
        ls_forbidden_cards = []

    # Get limited cards
    try:
        ls_limited_cards = [
            o_elem.get_attribute('alt') for o_elem in o_driver.find_element(By.CLASS_NAME, 'banlist').find_element(By.ID, 'limited').find_elements(By.CLASS_NAME, 'CardImages')
        ]
    except:
        ls_limited_cards = []

    # Get semi-limited cards
    try:
        ls_semilimited_cards = [
            o_elem.get_attribute('alt') for o_elem in o_driver.find_element(By.CLASS_NAME, 'banlist').find_element(By.ID, 'semi-limited').find_elements(By.CLASS_NAME, 'CardImages')
        ]
    except:
        ls_semilimited_cards = []

    # Format into dataframe
    d_banlist = {'Card': [], 'Limit': []}
    for s_card in ls_forbidden_cards:
        d_banlist['Card'].append(s_card)
        d_banlist['Limit'].append(0)
    for s_card in ls_limited_cards:
        d_banlist['Card'].append(s_card)
        d_banlist['Limit'].append(1)
    for s_card in ls_semilimited_cards:
        d_banlist['Card'].append(s_card)
        d_banlist['Limit'].append(2)
    df_banlist = pd.DataFrame(d_banlist)

    # Close webdriver
    o_driver.close()

    # Save
    df_banlist.to_csv(s_ban_list_file, index=False)


if __name__ == '__main__':
    main()
