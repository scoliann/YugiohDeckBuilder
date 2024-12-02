

# Do imports
import numpy as np
import pandas as pd
import collections as cl
import itertools
from tqdm import tqdm


# TODO:  Add GUI
# TODO:  Implement resuming from where you left off
# TODO:  Add a ban-also file
# TODO:  Add code to apply weights to each pillar


# Define global variables
SET_INVALID_PATHS = set()
SET_VALID_PATHS = set()
D_CARD_TO_GAME_STATE = {}
SET_DECK_LIST_SEEN = set()
D_CARD_FREQ_CNT_TO_BRICKLESS_FREQ = pd.read_pickle('d_card_freq_cnt_to_brickless_freq.pkl')


def fitness(df_card_pool, na_deck_list, i_path_size):

    # Determine deck list
    na_deck_list_idxs = np.where(na_deck_list)[0] // 3
    ls_deck_list = df_card_pool.index[na_deck_list_idxs].tolist()

    # Check permutations
    lt_valid_paths = []
    for t_path in itertools.chain.from_iterable(itertools.permutations(ls_deck_list, i) for i in range(2, i_path_size+1)):
        if t_path in SET_VALID_PATHS:
            lt_valid_paths.append(t_path)
        elif t_path in SET_INVALID_PATHS:
            continue
        else:
            b_valid = True
            for i in range(len(t_path)-1):
                if t_path[i: i+2] not in SET_VALID_PATHS:
                    b_valid = False
                    break
            if b_valid:
                lt_valid_paths.append(t_path)
                SET_VALID_PATHS.add(t_path)
            else:
                SET_INVALID_PATHS.add(t_path)
    lt_valid_paths = [tuple([s_card]) for s_card in ls_deck_list] + lt_valid_paths

    # Count paths to game states
    ls_game_states = []
    for t_path in lt_valid_paths:
        ls_game_states += D_CARD_TO_GAME_STATE[t_path[-1]]
    d_game_states = cl.Counter(ls_game_states)
    i_game_state_path_cnt = sum(d_game_states.values())
    i_game_state_path_cnt = np.power(i_game_state_path_cnt, 1.0 / i_path_size)

    # Get brickless frequency
    d_card_freq_cnt = cl.Counter(cl.Counter(na_deck_list_idxs).values())
    t_card_freq_cnt = (d_card_freq_cnt[1], d_card_freq_cnt[2], d_card_freq_cnt[3])
    f_brickless_freq = D_CARD_FREQ_CNT_TO_BRICKLESS_FREQ[t_card_freq_cnt]
    f_brickless_freq = round(f_brickless_freq, 2)

    # Create fitness array
    na_fitness = np.array([i_game_state_path_cnt, f_brickless_freq])

    # Return
    return na_fitness, d_game_states, lt_valid_paths


def optimize(df_banned_list, df_required_list, df_card_pool, i_deck_size, i_path_size, i_population, i_generations, f_mutation_rate, ls_input_deck_list=None, d_best_decks_data=None):

    # Create initial set of edges
    for s_card in df_card_pool.index:
        for s_card_supported in df_card_pool.columns[5:]:
            if df_card_pool.loc[s_card, s_card_supported]:
                SET_VALID_PATHS.add((s_card, s_card_supported))

    # Create mappings from cards to game states
    for s_card in df_card_pool.index:
        ts_game_states = df_card_pool.loc[s_card, df_card_pool.columns[:5]]
        ls_game_states = ts_game_states[ts_game_states == 1].index.tolist()
        D_CARD_TO_GAME_STATE[s_card] = ls_game_states
                
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
    ld_best_decks_valid_paths = np.array([np.nan], dtype=object) if d_best_decks_data is None else d_best_decks_data['valid_paths']     # This is for debugging
    for _ in tqdm(range(i_generations), desc='Deck Building'):

        # Evaluate fitnesses and update pareto frontier
        for na_deck_list in na_deck_lists:

            # If deck list has been seen before, skip
            t_deck_list = tuple(na_deck_list)
            if t_deck_list in SET_DECK_LIST_SEEN:
                continue

            # Get fitness
            na_fitness, d_path_term_cnt, lt_valid_paths = fitness(df_card_pool, na_deck_list, i_path_size)

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

                # Update valid paths
                ld_best_decks_valid_paths = ld_best_decks_valid_paths[np.logical_not(na_strictly_dominated_deck_idx_mask)]
                ld_best_decks_valid_paths = np.append(ld_best_decks_valid_paths, {'valid_paths': lt_valid_paths})

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
            na_child_1 = clean_up(na_child_0, i_deck_size)
            lna_children += [na_child_0, na_child_1]
        na_deck_lists = np.vstack([na_deck_lists, np.array(lna_children)])

    # Get deck lists
    lls_best_decks_eng = [df_card_pool.index[np.where(na_best_deck)[0] // 3].tolist() for na_best_deck in lna_best_decks]

    # Sort
    lt_best_decks_data = list(zip(lls_best_decks_eng, lna_best_decks, lna_best_decks_fitness, ld_best_decks_path_term_cnt, ld_best_decks_valid_paths))
    lt_best_decks_data = sorted(lt_best_decks_data, key=lambda x: x[2][0])
    lls_best_decks_eng, lna_best_decks, lna_best_decks_fitness, ld_best_decks_path_term_cnt, ld_best_decks_valid_paths = zip(*lt_best_decks_data)
    lls_best_decks_eng = list(lls_best_decks_eng)
    lna_best_decks = np.vstack(lna_best_decks)
    lna_best_decks_fitness = np.vstack(lna_best_decks_fitness)
    ld_best_decks_path_term_cnt = np.array(ld_best_decks_path_term_cnt)
    ld_best_decks_valid_paths = np.array(ld_best_decks_valid_paths)

    # Structure
    d_best_decks_data = {
        'deck_lists': lls_best_decks_eng,
        'deck_masks': lna_best_decks,
        'fitnesses': lna_best_decks_fitness,
        'term_cnt': ld_best_decks_path_term_cnt,
        'valid_paths': ld_best_decks_valid_paths,
    }

    # Return
    return d_best_decks_data


def main():

    # Define key variables
    s_banned_list_file = 'banned_list_goat.csv'
    s_required_list_file = 'required_list_goat.csv'
    s_card_pool_file = 'card_pool_chaos_control.csv'

    # Set random seeds
    np.random.seed(0)

    # Read in banned list
    df_banned_list = pd.read_csv(s_banned_list_file, index_col='Card')

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
    assert set(df_card_pool.values.flatten()) == set([0, 1]), \
        '\nError:\tMatrix contains one or more values that are not 0 or 1'

    # Get best decks data
    d_best_decks_data = optimize(
        df_banned_list=df_banned_list, 
        df_required_list=df_required_list, 
        df_card_pool=df_card_pool, 
        i_deck_size=40, 
        i_path_size=3, 
        i_population=4, 
        i_generations=500, 
        f_mutation_rate=0.05,
        ls_input_deck_list=['Asura Priest', 'Chaos Sorcerer', 'Cyber-Stein', 'Exiled Force', 'Fusilier Dragon, the Dual-Mode Beast', "Gravekeeper's Spy", 'Jinzo', 'Magical Merchant', 'Magician of Faith', 'Mystic Tomato', 'Night Assailant', 'Pyramid Turtle', 'Sangan', 'Shining Angel', 'Sinister Serpent', 'Skilled White Magician', 'Skilled White Magician', 'Spirit Reaper', 'Time Wizard', 'Tsukuyomi', 'Vampire Lord', 'Book of Life', 'Book of Life', 'Book of Moon', 'Brain Control', 'Brain Control', 'Card Destruction', 'Delinquent Duo', 'Heavy Storm', 'Metamorphosis', 'Mind Control', 'Monster Gate', 'Pot of Greed', 'Snatch Steal', 'Upstart Goblin', 'Deck Devastation Virus', 'Mirror Force', 'Phoenix Wing Wind Blast', 'Raigeki Break', 'Sakuretsu Armor'],
        d_best_decks_data=None,
    )


    import pdb; pdb.set_trace()



    import matplotlib.pyplot as plt

    # Example Nx2 numpy array
    points = d_best_decks_data['fitnesses']

    # Separate the array into x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Create the plot
    ###plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='tab:blue', marker='o', label='Decks')

    # Add labels, title, and legend
    plt.xlabel('Connectivity')
    plt.ylabel('Brickless Frequency')
    plt.title('Fitness Pareto Frontier')
    plt.legend()
    ###plt.grid(True)

    # Show the plot
    plt.show()


    # Get best deck data
    d_best_decks_data = optimize(
        df_banned_list=df_banned_list, 
        df_required_list=df_required_list, 
        df_card_pool=df_card_pool, 
        i_deck_size=40, 
        i_path_size=3, 
        i_population=4, 
        i_generations=500, 
        f_mutation_rate=0.05,
        ls_input_deck_list=None,
        d_best_decks_data=d_best_decks_data,
    )


    # Example Nx2 numpy array
    points = d_best_decks_data['fitnesses']

    # Separate the array into x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Create the plot
    ###plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='tab:blue', marker='o', label='Decks')

    # Add labels, title, and legend
    plt.xlabel('Connectivity')
    plt.ylabel('Brickless Frequency')
    plt.title('Fitness Pareto Frontier')
    plt.legend()
    ###plt.grid(True)

    # Show the plot
    plt.show()





    import pdb; pdb.set_trace()



    # Return
    #return ls_best_deck_list







    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()


