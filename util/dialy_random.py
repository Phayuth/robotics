import numpy as np

lunch = ["mcdonald",
         "burgerking",
         "momtouch",
         "vietnam food",
         "pasta",
         "yam yam",
         "japanese eel shop",
         "japanese shop 2nd floor",
         "ramen",
         "chicken spicy + rice",
         "korean pork belly and bean",
         "tobokki",
         ]

r = []
for i in range(len(lunch)):
    rand = np.random.rand()
    r.append(rand)

maxInd = np.argmax(np.array(r))
chosen = lunch[maxInd]
print(f"==>> chosen: \n{chosen}")