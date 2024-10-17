

# Do imports
import numpy as np
import pandas as pd
import collections as cl
from tqdm import tqdm
import itertools


# TODO:  Add option of putting in mandatory minimums for each pillar
# TODO:  Add code to check format of card pool file
# TODO:  Add code to enforce a minimum of each card type
# TODO:  Add code to insist on a certain balance between each pillar
# TODO:  Judge fitness simply by interconnectivity of card pool

# TODO:  Most interconnections, or most paths to a pillar?


# Define global variables
SET_INVALID_PATHS = set()
SET_VALID_PATHS = set()
D_DECK_TO_FITNESS = {}
D_CARD_TO_GAME_STATE = {}
D_CARD_FREQ_CNT_TO_BRICKLESS_FREQ = pd.read_pickle('d_card_freq_cnt_to_brickless_freq.pkl')


def fitness(df_card_pool, li_deck_list_idxs, i_path_size):

    # Check if deck fitness has been calculated
    lt_deck_list_idxs = tuple(li_deck_list_idxs)
    if lt_deck_list_idxs in D_DECK_TO_FITNESS:
        return D_DECK_TO_FITNESS[lt_deck_list_idxs], '', ''

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





    #print('\n\n--- am here ---')
    #import pdb; pdb.set_trace()




    # Calculate performance if N cards are invalidated  # TODO: Ehhhhh...
    d_card_in_path = cl.defaultdict(int)
    for t_path in lt_valid_paths:
        for s_card in set(t_path):
            d_card_in_path[s_card] += 1



    ##i_perf = len(lt_valid_paths) - max(d_card_in_path.values())
    ##i_perf = len(lt_valid_paths) * np.median(list(d_card_in_path.values()))
    ##i_perf = len(lt_valid_paths) * len(set(ls_deck_list))
    ##i_perf = len(lt_valid_paths) / np.mean(list(cl.Counter(ls_deck_list).values()))
    #i_perf = len(lt_valid_paths) * (1 - ((np.sum(np.array(cl.Counter(ls_deck_list).values()) == 3) * 3) / 40))
    #i_perf = len(lt_valid_paths)
    #if np.sum(np.array(list(cl.Counter(ls_deck_list).values())) == 3) > 3:
    #    i_perf = 0
    ##i_num_threes = np.sum(np.array(list(cl.Counter(ls_deck_list).values())) == 3)
    ##i_num_twos = np.sum(np.array(list(cl.Counter(ls_deck_list).values())) == 2)
    ##i_num_ones = np.sum(np.array(list(cl.Counter(ls_deck_list).values())) == 1)
    ##i_perf = np.log(len(lt_valid_paths)) * ((len(ls_deck_list) - i_num_threes) / len(ls_deck_list))
    ##i_perf = np.sqrt(len(lt_valid_paths)) * ((len(ls_deck_list) - i_num_threes) / len(ls_deck_list))        # TODO: This one's pretty good
    ##i_perf = (np.log(len(lt_valid_paths)) / np.log(i_path_size)) * ((len(ls_deck_list) - i_num_threes) / len(ls_deck_list))
    #i_perf = np.sqrt(len(lt_valid_paths)) / np.mean(list(cl.Counter(ls_deck_list).values()))
    #i_perf = np.sqrt(len(lt_valid_paths)) * (1 - (np.mean(list(cl.Counter(ls_deck_list).values())) / 3))
    #i_perf = np.sqrt(len(lt_valid_paths)) / ((i_num_threes * 3) + (i_num_twos * 2) + (i_num_ones * 1))
    #i_perf = len(lt_valid_paths) * ((len(ls_deck_list) - i_num_threes) / len(ls_deck_list))
    #i_perf = len(lt_valid_paths)
    

    # Count paths to game states
    ls_game_states = []
    for t_path in lt_valid_paths:
        ls_game_states += D_CARD_TO_GAME_STATE[t_path[-1]]
    d_game_states = cl.Counter(ls_game_states)

    # Get brickless frequency
    d_card_freq_cnt = cl.Counter(cl.Counter(li_deck_list_idxs).values())
    t_card_freq_cnt = (d_card_freq_cnt[1], d_card_freq_cnt[2], d_card_freq_cnt[3])
    f_brickless_freq = D_CARD_FREQ_CNT_TO_BRICKLESS_FREQ[t_card_freq_cnt]

    # Calcualte fitness
    ###i_perf = (np.log(len(lt_valid_paths)) / np.log(i_path_size)) * f_brickless_freq
    ###i_perf = np.power(len(lt_valid_paths), 1.0 / i_path_size) * f_brickless_freq
    i_perf = np.power(sum(d_game_states.values()), 1.0 / i_path_size) * f_brickless_freq
    ###i_perf = sum(d_game_states.values()) * f_brickless_freq
    ###i_perf = len(lt_valid_paths) * f_brickless_freq






    # Store fitness
    D_DECK_TO_FITNESS[lt_deck_list_idxs] = i_perf###i_valid_paths###sum(d_game_states.values())###


    # Return
    ###return i_valid_paths, d_game_states, lt_valid_paths
    return i_perf, d_game_states, lt_valid_paths


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
    na_best_deck = None
    f_best_deck_fitness = float('-inf')
    d_best_deck_path_term_cnt = None
    lt_best_valid_paths = None
    for _ in tqdm(range(i_generations), desc='Deck Building'):

        # Evaluate fitnesses
        lf_fitnesses = []
        ld_path_term_cnts = []
        llt_valid_paths = []
        for na_deck_list in na_deck_lists:
            na_deck_list_idxs = np.where(na_deck_list)[0] // 3
            f_fitness, d_path_term_cnt, lt_valid_paths = fitness(df_card_pool, na_deck_list_idxs, i_path_size)
            lf_fitnesses.append(f_fitness)
            ld_path_term_cnts.append(d_path_term_cnt)
            llt_valid_paths.append(lt_valid_paths)
        na_fitnesses = np.array(lf_fitnesses)

        # Update best deck
        i_best_deck_idx = np.argmax(na_fitnesses)
        if na_fitnesses[i_best_deck_idx] > f_best_deck_fitness:
            f_best_deck_fitness = na_fitnesses[i_best_deck_idx]
            d_best_deck_path_term_cnt = ld_path_term_cnts[i_best_deck_idx]
            na_best_deck = na_deck_lists[i_best_deck_idx]
            lt_best_valid_paths = llt_valid_paths[i_best_deck_idx]
        print('\n\n')
        print(f'Fitness:\t{f_best_deck_fitness}')
        print(f'Details:\t{d_best_deck_path_term_cnt}')

        # Remove half of population
        na_deck_lists = na_deck_lists[np.argsort(na_fitnesses)[len(na_fitnesses) // 2:]]

        # Apply crossover to create children, apply mutation to ensure deck size constraint
        def clean_up(na_child, i_deck_size):
            i_cards_needed = i_deck_size - np.sum(na_child)
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

    # Generate best deck list
    ls_best_deck_list = df_card_pool.index[np.where(na_best_deck)[0] // 3].tolist()

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

    # Get best deck list
    ls_best_deck_list = optimize(df_banned_list, df_required_list, df_card_pool, 40, 3, 8, 2000, 0.05)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()


