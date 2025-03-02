
# Do imports
import tkinter as tk
from tkinter import ttk
import threading
from deck_builder import read_in_data, optimize, plot_pareto_frontier, get_deck_image
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('Agg')

# Set global variables
S_PLOT_IMAGE = "gui/pareto_frontier_init.png"
S_DECK_IMAGE = "gui/deck_image_init.jpg"
D_BEST_DECKS_DATA = None
I_SELECTED_DECK = 0

def reload_plot_image():
    try:
        new_image = Image.open(S_PLOT_IMAGE)
        new_image = new_image.resize((573, 300))
        new_photo = ImageTk.PhotoImage(new_image)
        plot_image_label.config(image=new_photo)
        plot_image_label.image = new_photo
    except FileNotFoundError:
        print("Error: plot image not found.")

def reload_deck_image():
    try:
        new_image = Image.open(S_DECK_IMAGE)
        new_height = int(800 * (new_image.height / new_image.width))
        new_image = new_image.resize((800, new_height))
        new_photo = ImageTk.PhotoImage(new_image)
        deck_image_label.config(image=new_photo)
        deck_image_label.image = new_photo
    except FileNotFoundError:
        print("Error: deck image not found.")

def on_button_click():
    """Start the optimization in a separate thread."""
    try:
        total_iterations = int(entry_generations.get())
    except ValueError:
        print("Please enter a valid integer for generations!")
        return

    # Reset the progress bar
    progress_bar["value"] = 0
    progress_bar["maximum"] = total_iterations

    def progress_callback(progress):
        progress_bar["value"] = progress
        root.update_idletasks()

    def run_optimization():

        # Read in data
        df_banned_list, df_restricted_list, df_required_list, df_card_pool = read_in_data()

        # Specify global
        global S_PLOT_IMAGE
        global S_DECK_IMAGE
        global D_BEST_DECKS_DATA
        global I_SELECTED_DECK

        # Get parameter values
        try:
            f_plus_your_monsters = float(entry_plus_your_monsters.get())
            f_plus_your_hand = float(entry_plus_your_hand.get())
            f_minus_opponent_monsters = float(entry_minus_opponent_monsters.get())
            f_minus_opponent_hand = float(entry_minus_opponent_hand.get())
            f_minus_opponent_spell_trap = float(entry_minus_opponent_spell_trap.get())
        except ValueError:
            print("Please enter a valid integer for parameters!")
            return

        # Run optimization
        D_BEST_DECKS_DATA = optimize(
            df_banned_list=df_banned_list,
            df_restricted_list=df_restricted_list,
            df_required_list=df_required_list,
            df_card_pool=df_card_pool,
            i_deck_size=40,
            i_path_size=3,
            i_population=4,
            i_generations=total_iterations,
            f_mutation_rate=0.05,
            ls_input_deck_list=None,
            d_best_decks_data=D_BEST_DECKS_DATA,
            d_weights={
                'Plus Your Monsters': f_plus_your_monsters,
                'Plus Your Hand': f_plus_your_hand,
                'Minus Opponent Monsters': f_minus_opponent_monsters,
                'Minus Opponent Hand': f_minus_opponent_hand,
                'Minus Opponent Spell and Trap': f_minus_opponent_spell_trap,
            },
            fn_progress_callback=progress_callback,
        )
        print("Task completed!")

        # Create pareto frontier plot
        I_SELECTED_DECK = 0
        plot_pareto_frontier(D_BEST_DECKS_DATA, I_SELECTED_DECK)
        S_PLOT_IMAGE = "gui/pareto_frontier.png"

        # Get deck list
        ls_deck_list = D_BEST_DECKS_DATA['deck_lists'][I_SELECTED_DECK]
        S_DECK_IMAGE = "gui/deck_image.jpg"

        # Create deck image
        get_deck_image(ls_deck_list)
        
        # Reload the new image after optimization
        root.after(0, reload_plot_image)  # Schedule the reload_plot_image function to run on the main thread
        root.after(0, reload_deck_image)

    # Run optimizer
    threading.Thread(target=run_optimization).start()

def on_page_click(i_increment):

    # Specify global
    global I_SELECTED_DECK

    # Update plot
    if (S_PLOT_IMAGE == "gui/pareto_frontier.png") and (S_DECK_IMAGE == "gui/deck_image.jpg"):

        # Increment selected deck
        I_SELECTED_DECK = (I_SELECTED_DECK + i_increment) % len(D_BEST_DECKS_DATA['deck_lists'])

        # Create pareto frontier plot
        plot_pareto_frontier(D_BEST_DECKS_DATA, I_SELECTED_DECK)

        # Create deck image
        ls_deck_list = D_BEST_DECKS_DATA['deck_lists'][I_SELECTED_DECK]
        get_deck_image(ls_deck_list)

        # Reload images
        root.after(0, reload_plot_image)
        root.after(0, reload_deck_image)


root = tk.Tk()
root.title("Deck Building AI")
root.geometry("850x780")

# Create a main content frame to hold the image and parameters
content_frame = tk.Frame(root)
content_frame.pack(pady=0, padx=10, fill="both", expand=True)

# Populate with initial image
plot_image_label = tk.Label(content_frame)
plot_image_label.grid(row=0, column=0, padx=10)
plot_starting_image = Image.open(S_PLOT_IMAGE)
new_width = int(300 * (plot_starting_image.width / plot_starting_image.height))
plot_starting_image = plot_starting_image.resize((new_width, 300))
starting_photo = ImageTk.PhotoImage(plot_starting_image)
plot_image_label.config(image=starting_photo)
plot_image_label.image = starting_photo  # Keep a reference to avoid garbage collection

# Add a parameters frame for text labels, text boxes, and buttons
parameters_frame = tk.Frame(content_frame)
parameters_frame.grid(row=0, column=1, padx=10)

# Add "Generations" label and text box in row 0
label_generations = tk.Label(parameters_frame, text="Generations:")
label_generations.grid(row=0, column=0, padx=5, pady=5, sticky="e")
entry_generations = tk.Entry(parameters_frame, width=10)
entry_generations.insert(0, "50")
entry_generations.grid(row=0, column=1, padx=(0, 5), pady=5)

# Add parameter labels and text boxes
label_plus_your_monsters = tk.Label(parameters_frame, text="+ Your Monsters:")
label_plus_your_monsters.grid(row=1, column=0, padx=5, pady=5, sticky="e")
entry_plus_your_monsters = tk.Entry(parameters_frame, width=10)
entry_plus_your_monsters.insert(0, "1")
entry_plus_your_monsters.grid(row=1, column=1, padx=(0, 5), pady=5)

label_plus_your_hand = tk.Label(parameters_frame, text="+ Your Hand:")
label_plus_your_hand.grid(row=2, column=0, padx=5, pady=5, sticky="e")
entry_plus_your_hand = tk.Entry(parameters_frame, width=10)
entry_plus_your_hand.insert(0, "1")
entry_plus_your_hand.grid(row=2, column=1, padx=(0, 5), pady=5)

label_minus_opponent_monsters = tk.Label(parameters_frame, text="- Opponent Monsters:")
label_minus_opponent_monsters.grid(row=3, column=0, padx=5, pady=5, sticky="e")
entry_minus_opponent_monsters = tk.Entry(parameters_frame, width=10)
entry_minus_opponent_monsters.insert(0, "1")
entry_minus_opponent_monsters.grid(row=3, column=1, padx=(0, 5), pady=5)

label_minus_opponent_hand = tk.Label(parameters_frame, text="- Opponent Hand:")
label_minus_opponent_hand.grid(row=4, column=0, padx=5, pady=5, sticky="e")
entry_minus_opponent_hand = tk.Entry(parameters_frame, width=10)
entry_minus_opponent_hand.insert(0, "1")
entry_minus_opponent_hand.grid(row=4, column=1, padx=(0, 5), pady=5)

label_minus_opponent_spell_trap = tk.Label(parameters_frame, text="- Opponent Spell/Trap:")
label_minus_opponent_spell_trap.grid(row=5, column=0, padx=5, pady=5, sticky="e")
entry_minus_opponent_spell_trap = tk.Entry(parameters_frame, width=10)
entry_minus_opponent_spell_trap.insert(0, "1")
entry_minus_opponent_spell_trap.grid(row=5, column=1, padx=(0, 5), pady=5)

# Add progress bar below the parameter rows
progress_bar = ttk.Progressbar(parameters_frame, orient="horizontal", mode="determinate")
progress_bar.grid(row=6, column=0, columnspan=2, pady=(10, 10), sticky="ew")

# Add "Click Me!" button below the progress bar
button = tk.Button(parameters_frame, text="Click Me!", command=on_button_click)
button.grid(row=7, column=0, columnspan=2, pady=(10, 10))

# Create a new frame for paging
paging_frame = tk.Frame(parameters_frame)
paging_frame.grid(row=8, column=0, columnspan=2, pady=(5, 0), sticky="ew")

# Configure the columns in the buttons_frame for equal widths
paging_frame.columnconfigure(0, weight=1)
paging_frame.columnconfigure(1, weight=1)

# Determine the maximum button width (based on the text)
button_width = max(len("<-- Left"), len("Right -->"))

# Add "<-- Left" button
button_left = tk.Button(paging_frame, text="<-- Left", width=button_width, command=lambda: on_page_click(-1))
button_left.grid(row=0, column=0, padx=0, pady=0, sticky="ew")

# Add "Right -->" button
button_right = tk.Button(paging_frame, text="Right -->", width=button_width, command=lambda: on_page_click(1))
button_right.grid(row=0, column=1, padx=0, pady=0, sticky="ew")

# Create a main content frame to hold the image and parameters
results_frame = tk.Frame(root)
results_frame.pack(pady=0, padx=10, fill="both", expand=True)

# Populate with initial image
deck_image_label = tk.Label(results_frame)
deck_image_label.grid(row=0, column=0, padx=10)
deck_starting_image = Image.open(S_DECK_IMAGE)
new_height = int(800 * (deck_starting_image.height / deck_starting_image.width))
deck_starting_image = deck_starting_image.resize((800, new_height))
deck_starting_photo = ImageTk.PhotoImage(deck_starting_image)
deck_image_label.config(image=deck_starting_photo)
deck_image_label.image = deck_starting_photo  # Keep a reference to avoid garbage collection





'''
image_label.config(borderwidth=0, highlightthickness=0)
deck_image_label.config(borderwidth=0, highlightthickness=0)
'''

content_frame.config(bg="red")
results_frame.config(bg="blue")
plot_image_label.config(bg="green")
deck_image_label.config(bg="yellow")



root.mainloop()

