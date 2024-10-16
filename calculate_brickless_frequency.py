

# Do imports
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():

    # Define key variables
    i_deck_size = 40
    i_hand_size = 6
    i_iters = 10000
    s_data_file = 'd_card_freq_cnt_to_brickless_freq.pkl'

    # Compute all combinations of card frequency counts that can form a deck
    d_card_freq_cnt_to_deck = {}
    for i_3s in range(int(np.ceil(i_deck_size / 3.0)) + 1):
        for i_2s in range(int(np.ceil(i_deck_size / 2.0)) + 1):
            for i_1s in range(int(np.ceil(i_deck_size / 1.0)) + 1):

                # Check if total equals deck size
                i_total_cards = (i_3s * 3) + (i_2s * 2) + (i_1s * 1)
                if i_total_cards == i_deck_size:

                    # Create template deck list
                    na_1s_card_ids = np.arange(i_1s)
                    na_2s_card_ids = np.repeat(np.arange(i_2s) + i_1s, 2)
                    na_3s_card_ids = np.repeat(np.arange(i_3s) + i_1s + i_2s, 3)
                    na_card_ids = np.concatenate((na_1s_card_ids, na_2s_card_ids, na_3s_card_ids))

                    # Store
                    d_card_freq_cnt_to_deck[(i_1s, i_2s, i_3s)] = na_card_ids

    # Calculate frequency of brickless hands
    d_card_freq_cnt_to_brickless_freq = {}
    na_sample_idxs = np.argsort(np.random.random((i_iters, i_deck_size)), axis=1)[:, :i_hand_size]
    for t_card_freq_cnt in tqdm(d_card_freq_cnt_to_deck, desc='Monte Carlo'):

        # Get deck
        na_card_ids = d_card_freq_cnt_to_deck[t_card_freq_cnt]

        # Run monte carlo
        na_card_ids_sampled = na_card_ids[na_sample_idxs]
        na_brickless_hand = np.apply_along_axis(lambda row: len(set(row)) == len(row), axis=1, arr=na_card_ids_sampled)
        f_brickless_freq = np.mean(na_brickless_hand)

        # Store
        d_card_freq_cnt_to_brickless_freq[t_card_freq_cnt] = f_brickless_freq

    # Save
    pd.to_pickle(d_card_freq_cnt_to_brickless_freq, s_data_file)


if __name__ == '__main__':
    main()


