

# Do imports
import os
import re
import pandas as pd
import urllib.request
from tqdm import tqdm


# Do local imports
from args import *
from yugioh_metadata import *


def main():

    # Get directory name
    s_format_dir = s_card_pool_url.split('format=')[-1]

    # Define input and output files
    s_pickle_file = os.path.join(s_format_dir, 'card_pool_raw.pkl')
    s_csv_file = os.path.join(s_format_dir, 'card_pool.csv')

    # Read in dataframe
    df = pd.read_pickle(s_pickle_file)

    # Format columns
    df['tcg_release'] = df['tcg_release'].apply(lambda x: x.split(': ')[-1])
    df['category'] = df['category'].apply(lambda x: x.capitalize())
    df['attribute'] = df['attribute'].apply(lambda x: x.capitalize())
    df['level'] = df['level'].apply(lambda x: x.split()[-1])
    df['atk'] = df['atk'].apply(lambda x: '?' if x == '' else x.split()[-1])
    df['def'] = df['def'].apply(lambda x: '?' if x == '' else x.split()[-1])

    # Extract type data
    df = df.rename(columns={'type': 'type_data'})
    df['type_data'] = df['type_data'].apply(lambda x: x.split(' / '))

    def get_type(ts, d_types, ls_empty_case, b_return_lists):
        set_type_data = set(ts['type_data'])
        set_type_all = set(d_types[ts['category']])
        ls_types = sorted(list(set_type_data & set_type_all), reverse=True)
        if len(ls_types) == 0:
            ls_types = ls_empty_case
        if b_return_lists:
            return ls_types
        else:
            return ls_types[0]

    df['frame'] = df.apply(lambda ts: get_type(ts, {'Monster': monster_frame_types, 'Spell': spell_frame_types, 'Trap': trap_frame_types}, [ts['category']], True), axis=1)
    df['type'] = df.apply(lambda ts: get_type(ts, {'Monster': monster_types, 'Spell': spell_types, 'Trap': trap_types}, ['N/A'], False), axis=1)
    df['subtype'] = df.apply(lambda ts: get_type(ts, {'Monster': monster_subtypes, 'Spell': spell_subtypes, 'Trap': trap_subtypes}, ['N/A'], False), axis=1)

    # Re-organize columns
    df = df[['tcg_release', 'name', 'category', 'frame', 'type', 'subtype', 'level', 'atk', 'def', 'desc', 'img_url']]

    # Download images
    s_img_dir = os.path.join(s_format_dir, 'card_images')
    os.makedirs(s_img_dir, exist_ok=True)
    for idx, ts in tqdm(df.iterrows(), total=df.shape[0], desc='Downloading Card Images'):
        s_name = re.sub(r'[^a-zA-Z0-9 ]', '', ts['name']).replace(' ', '_')
        s_img_url = ts['img_url']
        s_img_path = os.path.join(s_img_dir, f'{s_name}.jpg')

        # Download the image
        urllib.request.urlretrieve(s_img_url, s_img_path)

        # Save image path
        df.loc[df.index[idx], 'img_path'] = s_img_path.split(os.sep, 1)[-1]

    # Save csv
    df.to_csv(s_csv_file, index=False)


if __name__ == '__main__':
    main()
