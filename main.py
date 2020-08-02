import pygame
import tkinter as tk
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model("saved_model")


def classify(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # convert to grayscale
    image = cv2.resize(image, (28, 28))              # resize to 28 by 28
    image = ~image                                   # invert image (b-w and w-b)
    image = image / 255                              # normalize image

    image = image.reshape(1, 28, 28, 1)              # reshape to pass into conv net
    image = tf.cast(image, tf.float64)               # typecast to allowed datatype

    prediction = model.predict(image)
    ans = np.argmax(prediction)
    confidence = round(np.max(prediction) * 100, 5)
    return ans, confidence


def popup(image):
    window = tk.Tk()
    window.geometry("200x200")
    window.title("Classifier")

    prediction, confidence = classify(image)
    label = tk.Label(window, text=f"The digit drawn is: {prediction} \n Confidence: {confidence}")
    label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    button = tk.Button(window, text="Draw Again!", command=window.destroy)
    button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    window.mainloop()


def gameloop():

    pygame.init()
    clock = pygame.time.Clock()

    screen_width = 150
    screen_height = 150
    radius = 10
    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill((255, 255, 255))

    pygame.display.set_caption("Digit Detector")
    icon = pygame.image.load("icon.png")
    pygame.display.set_icon(icon)

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    image = pygame.surfarray.array3d(screen).swapaxes(0, 1)
                    popup(image)
                    screen.fill((255, 255, 255))

            pressed = pygame.mouse.get_pressed()[0]

            if pressed and event.type == pygame.MOUSEMOTION:
                pygame.draw.circle(screen, (0, 0, 0), pygame.mouse.get_pos(), radius)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


gameloop()
