

# Do imports
import os
import cv2
import math
import numpy as np
import pandas as pd
import collections as cl
import itertools
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


# Do local imports
from args import *


# Define global variables
D_PATH_TO_GAME_STATE = {}
SET_DECK_LIST_SEEN = set()
D_CARD_FREQ_CNT_TO_BRICKLESS_FREQ = pd.read_pickle('d_card_freq_cnt_to_brickless_freq.pkl')


def read_in_data():

    # TODO:  Read in actual card pool and check cards in lists below against it

    # Define key variables
    s_banned_list_file = os.path.join(s_format_dir, 'ban_list.csv')

    # Read in banned list
    df_banned_list = pd.read_csv(s_banned_list_file, index_col='Card')

    # Read in restricted list
    df_restricted_list = pd.read_csv(s_restricted_list_file, index_col='Card')

    # Read in required list
    df_required_list = pd.read_csv(s_required_list_file, index_col='Card')

    # Read in card pool
    df_card_pool = pd.read_csv(s_card_pool_file, index_col='Card Pool')

    # Check format of card pool file
    ls_game_states = ['Plus Your Monsters', 'Plus Your Hand', 'Minus Opponent Monsters', 'Minus Opponent Spell and Trap', 'Minus Opponent Hand']
    assert df_card_pool.columns.tolist()[:5] == ls_game_states, \
        f'\nError:\tOne of the following columns is missing: {ls_game_states}'
    assert df_card_pool.columns.tolist()[5:] == df_card_pool.index.tolist(), \
        '\nError:\tCard name columns do not match card name indices'

    # Return
    return df_banned_list, df_restricted_list, df_required_list, df_card_pool


def fitness(df_card_pool, na_deck_list, i_path_size, d_weights):

    # Determine deck list
    na_deck_list_idxs = np.where(na_deck_list)[0] // 3
    ls_deck_list = df_card_pool.index[na_deck_list_idxs].tolist()

    # Calculate factorial
    i_path_size_fact = math.factorial(i_path_size)

    # Define a generator to iterate over paths to game states
    def generate_paths(ls_deck_list, i_path_size, ls_game_states):

        # Generate over combinations
        for t_path, s_game_state in itertools.product(
            itertools.combinations(ls_deck_list, i_path_size),
            ls_game_states,
        ):

            # Create single path
            t_path = t_path + (s_game_state,)

            # Yield
            yield t_path

    # Iterate over all paths to game states
    d_game_states = cl.defaultdict(lambda: 0.0)
    for t_path_to_game_state in generate_paths(ls_deck_list, i_path_size, df_card_pool.columns[:5]):

        # Compute fitness
        if t_path_to_game_state not in D_PATH_TO_GAME_STATE:

            # Compute fitness of new path
            f_fitness = 1.0
            for i in range(len(t_path_to_game_state)-1):
                s_node = t_path_to_game_state[i]
                s_node_next = t_path_to_game_state[i+1]
                f_pct = df_card_pool.at[s_node, s_node_next]
                f_fitness *= f_pct

            # Multiply by number of ways the combination can be made
            #   Note:  This approach works because nPk = nCk * kPk = nCk * k!
            f_fitness *= i_path_size_fact

            # Store result
            D_PATH_TO_GAME_STATE[t_path_to_game_state] = f_fitness
        
        # Update
        d_game_states[t_path_to_game_state[-1]] += D_PATH_TO_GAME_STATE[t_path_to_game_state]

    # Calculate weighted connectivity to game states
    d_weights = cl.defaultdict(lambda: 1.0) if d_weights is None else cl.defaultdict(lambda: 0.0, d_weights)
    f_game_state_path_val = sum(d_game_states[k] * d_weights[k] for k in d_game_states)
    f_game_state_path_val = np.power(f_game_state_path_val, 1.0 / i_path_size)

    # Get brickless frequency
    d_card_freq_cnt = cl.Counter(cl.Counter(na_deck_list_idxs).values())
    t_card_freq_cnt = (d_card_freq_cnt[1], d_card_freq_cnt[2], d_card_freq_cnt[3])
    f_brickless_freq = D_CARD_FREQ_CNT_TO_BRICKLESS_FREQ[t_card_freq_cnt]

    # Create fitness array
    na_fitness = np.array([f_game_state_path_val, f_brickless_freq])

    # Return
    return na_fitness, dict(d_game_states)


def optimize(df_banned_list, df_restricted_list, df_required_list, df_card_pool, i_deck_size, i_path_size, i_population, i_generations, f_mutation_rate, 
             ls_input_deck_list=None, d_best_decks_data=None, d_weights=None, fn_progress_callback=None):

    # Create unified banned list
    df_banned_list = pd.concat([df_banned_list, df_restricted_list])
                
    # Create a vector to enforce the banned list
    li_banned_list = []
    for s_card in df_card_pool.index:
        if s_card in df_banned_list.index:
            li_copies = [0, 0, 0]
            for i_idx in range(df_banned_list.loc[s_card, 'Limit']):
                li_copies[i_idx] = 1
        else:
            li_copies = [1, 1, 1]
        li_banned_list += li_copies
    na_banned_list = np.array(li_banned_list)

    # Create a vector to enforce the required list
    li_required_list = []
    for s_card in df_card_pool.index:
        if s_card in df_required_list.index:
            li_copies = [0, 0, 0]
            for i_idx in range(df_required_list.loc[s_card, 'Limit']):
                li_copies[i_idx] = 1
        else:
            li_copies = [0, 0, 0]
        li_required_list += li_copies
    na_required_list = np.array(li_required_list)

    # Create a vector for input deck list
    if ls_input_deck_list:  
        d_input_deck_card_cnts = cl.Counter(ls_input_deck_list)
        li_input_deck_list = []
        for s_card in df_card_pool.index:
            if s_card in d_input_deck_card_cnts:
                li_copies = [0, 0, 0]
                for i_idx in range(d_input_deck_card_cnts[s_card]):
                    li_copies[i_idx] = 1
            else:
                li_copies = [0, 0, 0]
            li_input_deck_list += li_copies
        lna_input_deck_list = [np.array(li_input_deck_list)]
    else:
        lna_input_deck_list = []

    # Create deck list feature vectors
    li_deck_lists = [] + lna_input_deck_list
    for _ in range(i_population - len(li_deck_lists)):
        na_deck_list_idxs_required = np.where(na_required_list)[0]
        na_deck_list_idxs_optional = np.random.choice(np.where(na_banned_list - na_required_list)[0], size=i_deck_size - sum(na_required_list), replace=False)
        na_deck_list_idxs = np.concat((na_deck_list_idxs_required, na_deck_list_idxs_optional))
        na_deck_list = np.zeros_like(na_banned_list)
        na_deck_list[na_deck_list_idxs] = 1
        li_deck_lists.append(na_deck_list)
    na_deck_lists = np.array(li_deck_lists)

    # Evolve over generations
    lna_best_decks = np.full((1, na_deck_lists.shape[1]), np.nan) if d_best_decks_data is None else d_best_decks_data['deck_masks']
    lna_best_decks_fitness = np.array([[-np.inf, -np.inf]]) if d_best_decks_data is None else d_best_decks_data['fitnesses']
    ld_best_decks_path_term_cnt = np.array([np.nan], dtype=object) if d_best_decks_data is None else d_best_decks_data['term_cnt']      # This is for debugging
    for i_gen in tqdm(range(i_generations), desc='Deck Building'):

        # Update GUI progress bar
        if fn_progress_callback is not None:
            fn_progress_callback(i_gen + 1)

        # Evaluate fitnesses and update pareto frontier
        for na_deck_list in na_deck_lists:

            # If deck list has been seen before, skip
            t_deck_list = tuple(na_deck_list)
            if t_deck_list in SET_DECK_LIST_SEEN:
                print('\n\nSEEN!!!\n\n')
                continue

            # Get fitness
            na_fitness, d_path_term_cnt = fitness(df_card_pool, na_deck_list, i_path_size, d_weights)

            # Update pareto frontier
            b_dominates_a_deck = np.any(np.any(na_fitness > lna_best_decks_fitness, axis=1))
            b_dominated_by_a_deck = np.any(np.all(na_fitness <= lna_best_decks_fitness, axis=1))
            na_strictly_dominated_deck_idx_mask = np.all(na_fitness >= lna_best_decks_fitness, axis=1)
            if b_dominates_a_deck and not b_dominated_by_a_deck:

                # Update fitnesses
                lna_best_decks_fitness = lna_best_decks_fitness[np.logical_not(na_strictly_dominated_deck_idx_mask), :]
                lna_best_decks_fitness = np.vstack([lna_best_decks_fitness, na_fitness])

                # Update deck lists
                lna_best_decks = lna_best_decks[np.logical_not(na_strictly_dominated_deck_idx_mask), :]
                lna_best_decks = np.vstack([lna_best_decks, na_deck_list])

                # Update path counts
                ld_best_decks_path_term_cnt = ld_best_decks_path_term_cnt[np.logical_not(na_strictly_dominated_deck_idx_mask)]
                ld_best_decks_path_term_cnt = np.append(ld_best_decks_path_term_cnt, d_path_term_cnt)

                # Calculate deck that would be selected
                if True:
                    na_best_decks_fitness = np.prod(lna_best_decks_fitness, axis=1)
                    na_best_deck_fitness = lna_best_decks_fitness[np.argmax(na_best_decks_fitness)]
                    d_best_deck_path_term_cnt = ld_best_decks_path_term_cnt[np.argmax(na_best_decks_fitness)]
                    print('\n\n')
                    print(f'Fitness #1:\t{np.max(na_best_decks_fitness)}')
                    print(f'Fitness #2:\t{list(na_best_deck_fitness)}')
                    print(f'Details:\t{d_best_deck_path_term_cnt}')
                    print(lna_best_decks_fitness.shape) 

            # Mark as seen
            SET_DECK_LIST_SEEN.add(t_deck_list)  

        # Choose a new parent population
        na_deck_lists = lna_best_decks[np.random.choice(lna_best_decks.shape[0], size=int(na_deck_lists.shape[0] / 2), replace=True)]

        # Apply crossover to create children, apply mutation to ensure deck size constraint
        def clean_up(na_child, i_deck_size):
            i_cards_needed = int(i_deck_size - np.sum(na_child))
            if i_cards_needed < 0:
                na_remove_card_idxs = np.random.choice(np.where(na_child - na_required_list)[0], size=-i_cards_needed, replace=False)
                na_child[na_remove_card_idxs] = 0
            elif i_cards_needed > 0:
                na_potential_card_list = np.copy(na_banned_list)
                na_potential_card_list[np.where(na_child)[0]] = 0
                na_add_card_idxs = np.random.choice(np.where(na_potential_card_list)[0], size=i_cards_needed, replace=False)
                na_child[na_add_card_idxs] = 1
            return na_child
        lna_children = []
        na_deck_lists = na_deck_lists[np.random.choice(na_deck_lists.shape[0], size=na_deck_lists.shape[0], replace=False)]
        for i_idx in range(0, na_deck_lists.shape[0], 2):
            na_parent_0 = na_deck_lists[i_idx]
            na_parent_1 = na_deck_lists[i_idx + 1]
            na_crossover_idxs = np.random.choice(na_deck_lists.shape[1], size=2, replace=False)
            i_crossover_idx_min = min(na_crossover_idxs)
            i_crossover_idx_max = max(na_crossover_idxs)
            na_child_0 = np.concat([na_parent_0[:i_crossover_idx_min], na_parent_1[i_crossover_idx_min: i_crossover_idx_max], na_parent_0[i_crossover_idx_max:]])
            na_child_1 = np.concat([na_parent_1[:i_crossover_idx_min], na_parent_0[i_crossover_idx_min: i_crossover_idx_max], na_parent_1[i_crossover_idx_max:]])
            na_child_0 = np.abs(na_child_0 - ((np.random.random(len(na_child_0)) < f_mutation_rate) * (na_banned_list - na_required_list)))
            na_child_1 = np.abs(na_child_1 - ((np.random.random(len(na_child_1)) < f_mutation_rate) * (na_banned_list - na_required_list)))
            na_child_0 = clean_up(na_child_0, i_deck_size)
            na_child_1 = clean_up(na_child_1, i_deck_size)
            lna_children += [na_child_0, na_child_1]
        na_deck_lists = np.vstack([na_deck_lists, np.array(lna_children)])

    # Get deck lists
    lls_best_decks_eng = [df_card_pool.index[np.where(na_best_deck)[0] // 3].tolist() for na_best_deck in lna_best_decks]

    # Sort
    lt_best_decks_data = list(zip(lls_best_decks_eng, lna_best_decks, lna_best_decks_fitness, ld_best_decks_path_term_cnt))
    lt_best_decks_data = sorted(lt_best_decks_data, key=lambda x: x[2][0])
    lls_best_decks_eng, lna_best_decks, lna_best_decks_fitness, ld_best_decks_path_term_cnt = zip(*lt_best_decks_data)
    lls_best_decks_eng = list(lls_best_decks_eng)
    lna_best_decks = np.vstack(lna_best_decks)
    lna_best_decks_fitness = np.vstack(lna_best_decks_fitness)
    ld_best_decks_path_term_cnt = np.array(ld_best_decks_path_term_cnt)

    # Structure
    d_best_decks_data = {
        'deck_lists': lls_best_decks_eng,
        'deck_masks': lna_best_decks,
        'fitnesses': lna_best_decks_fitness,
        'term_cnt': ld_best_decks_path_term_cnt,
    }

    # Return
    return d_best_decks_data


def plot_pareto_frontier(d_best_decks_data, i_selected_deck=None):

    # Get multi-dimensional fitness values
    points = d_best_decks_data['fitnesses']

    # Separate individual fitness values into x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Create the plot
    plt.plot(x, y, color='tab:blue', marker='o', zorder=1)

    # Highlight selected deck
    if i_selected_deck is not None:
        i_deck_idx = i_selected_deck % len(x)
        plt.scatter(x[i_deck_idx], y[i_deck_idx], color='fuchsia', s=80, zorder=2)

    # Add labels, title, and legend
    plt.xlabel('Connectivity')
    plt.ylabel('Brickless Frequency')
    plt.title('Fitness Pareto Frontier')

    # Show the plot
    plt.savefig('gui/pareto_frontier.png')

    # Close
    plt.close()


def get_deck_image(ls_cards, i_cards_per_row=10):

    # Read into dataframe
    df = pd.read_csv(os.path.join(s_format_dir, 'card_pool.csv'))

    # Organize card order
    ls_monsters = []
    ls_spells = []
    ls_traps = []
    for s_card in sorted(ls_cards):
        ts_row = df[df['name'] == s_card].iloc[0]
        s_category = ts_row['category']
        s_img_path = ts_row['img_path']
        if s_category == 'Monster':
            ls_monsters.append((s_card, s_img_path))
        elif s_category == 'Spell':
            ls_spells.append((s_card, s_img_path))
        elif s_category == 'Trap':
            ls_traps.append((s_card, s_img_path))      
    ls_cards, ls_img_paths = zip(*(ls_spells + ls_monsters + ls_traps))

    # Divide into rows
    lls_img_paths = [[]]
    for s_img_path in ls_img_paths:
        if len(lls_img_paths[-1]) < i_cards_per_row:
            lls_img_paths[-1].append(s_img_path)
        else:
            lls_img_paths.append([s_img_path])

    # Construct image
    lna_rows = []
    for i, ls_img_paths in enumerate(lls_img_paths):
        lna_imgs = [cv2.imread(os.path.join(s_format_dir, s_img_path)) for s_img_path in ls_img_paths]
        na_row = np.hstack(lna_imgs)
        if lna_rows and lna_rows[-1].shape[1] > na_row.shape[1]:
            na_filler = np.zeros((na_row.shape[0], lna_rows[-1].shape[1] - na_row.shape[1], 3))
            na_row = np.hstack([na_row, na_filler])
        lna_rows.append(na_row)
    na_deck_img = np.vstack(lna_rows)
    cv2.imwrite('gui/deck_image.jpg', na_deck_img)


def main():

    # Read in data
    df_banned_list, df_restricted_list, df_required_list, df_card_pool = read_in_data()

    # Get best decks data
    d_best_decks_data = optimize(
        df_banned_list=df_banned_list,
        df_restricted_list=df_restricted_list,
        df_required_list=df_required_list, 
        df_card_pool=df_card_pool, 
        i_deck_size=40, 
        i_path_size=3, 
        i_population=4, 
        i_generations=500, 
        f_mutation_rate=0.05,
        ls_input_deck_list=None,
        d_best_decks_data=None,
        d_weights={'Plus Your Monsters': 1, 'Plus Your Hand': 1, 'Minus Opponent Monsters': 1, 'Minus Opponent Spell and Trap': 1, 'Minus Opponent Hand': 1},
        fn_progress_callback=None,
    )

    # Create plot
    plot_pareto_frontier(d_best_decks_data)


    import pdb; pdb.set_trace()



if __name__ == '__main__':
    main()


