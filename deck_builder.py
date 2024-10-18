

# Do imports
import numpy as np
import pandas as pd
import collections as cl
import itertools
from tqdm import tqdm


# TODO:  Add code to apply weights to each pillar


# Define global variables
SET_INVALID_PATHS = set()
SET_VALID_PATHS = set()
D_CARD_TO_GAME_STATE = {}
SET_DECK_LIST_SEEN = set()
D_CARD_FREQ_CNT_TO_BRICKLESS_FREQ = pd.read_pickle('d_card_freq_cnt_to_brickless_freq.pkl')


def fitness(df_card_pool, li_deck_list_idxs, i_path_size):

    # Define key variables
    ls_deck_list = df_card_pool.index[li_deck_list_idxs].tolist()

    # Check permutations
    i_valid_paths = 0
    lt_valid_paths = []
    for t_path in itertools.chain.from_iterable(itertools.permutations(ls_deck_list, i) for i in range(2, i_path_size+1)):
        if t_path in SET_VALID_PATHS:
            i_valid_paths += 1
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
    d_card_freq_cnt = cl.Counter(cl.Counter(li_deck_list_idxs).values())
    t_card_freq_cnt = (d_card_freq_cnt[1], d_card_freq_cnt[2], d_card_freq_cnt[3])
    f_brickless_freq = D_CARD_FREQ_CNT_TO_BRICKLESS_FREQ[t_card_freq_cnt]
    f_brickless_freq = round(f_brickless_freq, 2)

    # Create fitness array
    na_fitness = np.array([i_game_state_path_cnt, f_brickless_freq])

    # Return
    return na_fitness, d_game_states, lt_valid_paths


def optimize(df_banned_list, df_required_list, df_card_pool, i_deck_size, i_path_size, i_population, i_generations, f_mutation_rate):

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

    # Create deck list feature vectors
    li_deck_lists = []
    for _ in range(i_population):
        na_deck_list_idxs_required = np.where(na_required_list)[0]
        na_deck_list_idxs_optional = np.random.choice(np.where(na_banned_list - na_required_list)[0], size=i_deck_size - sum(na_required_list), replace=False)
        na_deck_list_idxs = np.concat((na_deck_list_idxs_required, na_deck_list_idxs_optional))
        na_deck_list = np.zeros_like(na_banned_list)
        na_deck_list[na_deck_list_idxs] = 1
        li_deck_lists.append(na_deck_list)
    na_deck_lists = np.array(li_deck_lists)

    # Evolve over generations
    lna_best_decks = np.full((1, na_deck_lists.shape[1]), np.nan)
    lna_best_decks_fitness = np.array([[-np.inf, -np.inf]])
    ld_best_decks_path_term_cnt = np.array([np.nan], dtype=object)            # This is for debugging
    ldlt_best_valid_paths = np.array([np.nan], dtype=object)                  # This is for debugging
    for _ in tqdm(range(i_generations), desc='Deck Building'):

        # Evaluate fitnesses and update pareto frontier
        for na_deck_list in na_deck_lists:

            # If deck list has been seen before, skip
            t_deck_list = tuple(na_deck_list)
            if t_deck_list in SET_DECK_LIST_SEEN:
                continue

            # Get fitness
            na_deck_list_idxs = np.where(na_deck_list)[0] // 3
            na_fitness, d_path_term_cnt, lt_valid_paths = fitness(df_card_pool, na_deck_list_idxs, i_path_size)

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

                ldlt_best_valid_paths = ldlt_best_valid_paths[np.logical_not(na_strictly_dominated_deck_idx_mask)]
                ldlt_best_valid_paths = np.append(ldlt_best_valid_paths, {'valid_paths': lt_valid_paths})

            # Mark as seen
            SET_DECK_LIST_SEEN.add(t_deck_list)

        # Calculate deck that would be selected
        if True:
            na_best_decks_fitness = np.prod(lna_best_decks_fitness, axis=1)
            na_best_deck_fitness = lna_best_decks_fitness[np.argmax(na_best_decks_fitness)]
            d_best_deck_path_term_cnt = ld_best_decks_path_term_cnt[np.argmax(na_best_decks_fitness)]
            print('\n\n')
            print(f'Fitness:\t{list(na_best_deck_fitness)}')
            print(f'Details:\t{d_best_deck_path_term_cnt}')
            print(lna_best_decks_fitness.shape)       

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

    # Calculate deck that would be selected
    na_best_decks_fitness = np.prod(lna_best_decks_fitness, axis=1)
    na_best_deck = lna_best_decks[np.argmax(na_best_decks_fitness)]
    na_best_deck_fitness = lna_best_decks_fitness[np.argmax(na_best_decks_fitness)]
    d_best_deck_path_term_cnt = ld_best_decks_path_term_cnt[np.argmax(na_best_decks_fitness)]
    print('\n\n')
    print(f'Fitness:\t{list(na_best_deck_fitness)}')
    print(f'Details:\t{d_best_deck_path_term_cnt}')
    print(lna_best_decks_fitness.shape)  

    # Generate best deck list
    ls_best_deck_list = df_card_pool.index[np.where(na_best_deck)[0] // 3].tolist()


    print('\n\n--- at end ---')
    import pdb; pdb.set_trace()


    # Check that deck adheres to banned list
    for s_card, i_cnt in cl.Counter(ls_best_deck_list).items():
        if s_card in df_banned_list.index:
            i_cnt_max = df_banned_list.loc[s_card, 'Limit']
            assert i_cnt <= i_cnt_max, \
                f'\n\nError:\tCard {s_card} has {i_cnt} copies when the banned list permits {i_cnt_max}'

    # Return
    return ls_best_deck_list


def main():

    # Define key variables
    s_banned_list_file = 'banned_list_goat.csv'
    s_required_list_file = 'required_list_goat.csv'
    s_card_pool_file = 'card_pool_chaos_control.csv'

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

    # Get best deck list
    ls_best_deck_list = optimize(df_banned_list, df_required_list, df_card_pool, 40, 3, 4, 10000, 0.05)


    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()


