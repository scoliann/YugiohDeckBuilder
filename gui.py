import tkinter as tk
from tkinter import ttk
import threading
from deck_builder import read_in_data, optimize, plot_pareto_frontier
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend


# Set global variables
D_BEST_DECKS_DATA = None


def on_button_click():
    """Start the optimization in a separate thread."""
    try:
        total_iterations = int(text_box.get())
    except ValueError:
        print("Please enter a valid integer!")
        return

    # Reset the progress bar
    progress_bar["value"] = 0
    progress_bar["maximum"] = total_iterations

    def progress_callback(progress):
        progress_bar["value"] = progress
        root.update_idletasks()  # Update the GUI

    def reload_image():
        try:
            new_image = Image.open("pareto_frontier.png")
            new_image = new_image.resize((600, 400))  # Resize image for clarity
            new_photo = ImageTk.PhotoImage(new_image)
            image_label.config(image=new_photo)
            image_label.image = new_photo  # Keep a reference to avoid garbage collection
            print("Image updated!")
        except FileNotFoundError:
            print("Error: pareto_frontier.png not found after optimization.")

    def run_optimization():

        # Read in data
        df_banned_list, df_restricted_list, df_required_list, df_card_pool = read_in_data()

        # Specify global
        global D_BEST_DECKS_DATA

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
            d_weights={'Plus Your Monsters': 1, 'Plus Your Hand': 1, 'Minus Opponent Monsters': 1, 'Minus Opponent Spell and Trap': 1, 'Minus Opponent Hand': 4},
            fn_progress_callback=progress_callback,
        )
        print("Task completed!")

        # Create pareto frontier plot
        plot_pareto_frontier(D_BEST_DECKS_DATA)

        # Reload the new image after optimization
        root.after(0, reload_image)  # Schedule the reload_image function to run on the main thread

    threading.Thread(target=run_optimization).start()

root = tk.Tk()
root.title("Simple GUI with Progress Bar")
root.geometry("800x600")  # Width=800, Height=600

frame = tk.Frame(root)
frame.pack(pady=20, padx=20)

label = tk.Label(frame, text="Generations:")
label.grid(row=0, column=0, padx=5)

text_box = tk.Entry(frame)
text_box.insert(0, "50")
text_box.grid(row=0, column=1, padx=5)

button = tk.Button(frame, text="Click Me!", command=on_button_click)
button.grid(row=0, column=2, padx=5)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
progress_bar.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

root.mainloop()

