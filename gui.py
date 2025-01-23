import tkinter as tk
from tkinter import ttk
import threading
from deck_builder import read_in_data, optimize, plot_pareto_frontier
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

# Set global variables
S_IMAGE = "start_image.png"
D_BEST_DECKS_DATA = None
I_SELECTED_DECK = 0

def reload_image():
    try:
        new_image = Image.open(S_IMAGE)
        new_image = new_image.resize((600, 400))  # Resize image for clarity
        new_photo = ImageTk.PhotoImage(new_image)
        image_label.config(image=new_photo)
        image_label.image = new_photo  # Keep a reference to avoid garbage collection
        print("Image updated!")
    except FileNotFoundError:
        print("Error: image not found.")

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
        root.update_idletasks()  # Update the GUI

    def run_optimization():
        # Read in data
        df_banned_list, df_restricted_list, df_required_list, df_card_pool = read_in_data()

        # Specify global
        global S_IMAGE
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
        S_IMAGE = "pareto_frontier.png"

        # Reload the new image after optimization
        root.after(0, reload_image)  # Schedule the reload_image function to run on the main thread

    threading.Thread(target=run_optimization).start()

def on_left_click():

    # Specify global
    global I_SELECTED_DECK

    # Update
    if S_IMAGE == "pareto_frontier.png":

        # Increment selected deck
        I_SELECTED_DECK -= 1

        # Create pareto frontier plot
        plot_pareto_frontier(D_BEST_DECKS_DATA, I_SELECTED_DECK)

        # Reload the new image after optimization
        root.after(0, reload_image)  # Schedule the reload_image function to run on the main thread

def on_right_click():

    # Specify global
    global I_SELECTED_DECK

    # Update
    if S_IMAGE == "pareto_frontier.png":

        # Increment selected deck
        I_SELECTED_DECK += 1

        # Create pareto frontier plot
        plot_pareto_frontier(D_BEST_DECKS_DATA, I_SELECTED_DECK)

        # Reload the new image after optimization
        root.after(0, reload_image)  # Schedule the reload_image function to run on the main thread

root = tk.Tk()
root.title("Deck Building AI")
root.geometry("925x650")

# Create a main content frame to hold the image and parameters
content_frame = tk.Frame(root)
content_frame.pack(pady=10, padx=10, fill="both", expand=True)

# Populate with initial image
image_label = tk.Label(content_frame)
image_label.grid(row=0, column=0, padx=10)
starting_image = Image.open(S_IMAGE)
new_width = int(400 * (starting_image.width / starting_image.height))
starting_image = starting_image.resize((new_width, 400))
left = (new_width - 600) // 2
right = left + 600
top = 0
bottom = 400
starting_image = starting_image.crop((left, top, right, bottom))
starting_photo = ImageTk.PhotoImage(starting_image)
image_label.config(image=starting_photo)
image_label.image = starting_photo  # Keep a reference to avoid garbage collection

# Add a parameters frame for text labels, text boxes, and buttons
parameters_frame = tk.Frame(content_frame)
parameters_frame.grid(row=0, column=1, padx=10)

# Add "Generations" label and text box in row 0
label_generations = tk.Label(parameters_frame, text="Generations:")
label_generations.grid(row=0, column=0, padx=5, pady=5, sticky="e")
entry_generations = tk.Entry(parameters_frame)
entry_generations.insert(0, "50")
entry_generations.grid(row=0, column=1, pady=5)

# Add parameter labels and text boxes
label_plus_your_monsters = tk.Label(parameters_frame, text="+ Your Monsters:")
label_plus_your_monsters.grid(row=1, column=0, padx=5, pady=5, sticky="e")
entry_plus_your_monsters = tk.Entry(parameters_frame)
entry_plus_your_monsters.insert(0, "1")
entry_plus_your_monsters.grid(row=1, column=1, pady=5)

label_plus_your_hand = tk.Label(parameters_frame, text="+ Your Hand:")
label_plus_your_hand.grid(row=2, column=0, padx=5, pady=5, sticky="e")
entry_plus_your_hand = tk.Entry(parameters_frame)
entry_plus_your_hand.insert(0, "1")
entry_plus_your_hand.grid(row=2, column=1, pady=5)

label_minus_opponent_monsters = tk.Label(parameters_frame, text="- Opponent Monsters:")
label_minus_opponent_monsters.grid(row=3, column=0, padx=5, pady=5, sticky="e")
entry_minus_opponent_monsters = tk.Entry(parameters_frame)
entry_minus_opponent_monsters.insert(0, "1")
entry_minus_opponent_monsters.grid(row=3, column=1, pady=5)

label_minus_opponent_hand = tk.Label(parameters_frame, text="- Opponent Hand:")
label_minus_opponent_hand.grid(row=4, column=0, padx=5, pady=5, sticky="e")
entry_minus_opponent_hand = tk.Entry(parameters_frame)
entry_minus_opponent_hand.insert(0, "1")
entry_minus_opponent_hand.grid(row=4, column=1, pady=5)

label_minus_opponent_spell_trap = tk.Label(parameters_frame, text="- Opponent Spell/Trap:")
label_minus_opponent_spell_trap.grid(row=5, column=0, padx=5, pady=5, sticky="e")
entry_minus_opponent_spell_trap = tk.Entry(parameters_frame)
entry_minus_opponent_spell_trap.insert(0, "1")
entry_minus_opponent_spell_trap.grid(row=5, column=1, pady=5)

# Add progress bar below the parameter rows
progress_bar = ttk.Progressbar(parameters_frame, orient="horizontal", mode="determinate")
progress_bar.grid(row=6, column=0, columnspan=2, pady=(55, 10), sticky="ew")

# Add "Click Me!" button below the progress bar
button = tk.Button(parameters_frame, text="Click Me!", command=on_button_click)
button.grid(row=7, column=0, columnspan=2, pady=(10, 55))

# Add "<-- Left" button
button_left = tk.Button(parameters_frame, text="<-- Left", command=on_left_click)
button_left.grid(row=9, column=0, pady=5, sticky="ew")

# Add "Right -->" button
button_right = tk.Button(parameters_frame, text="Right -->", command=on_right_click)
button_right.grid(row=9, column=1, pady=5, sticky="ew")

root.mainloop()

