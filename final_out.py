from category_style import predict
from Outfit_recommedation import recommedation_model
import os

#Assigning input path
img_path = input("Enter image path:")
image = os.listdir(img_path)

# This loop will help in writing the output in text file
with open("output.txt", "w") as file:
    for i in image[:50]:
        #predict the style category for an image
        data = predict.predict(f"{img_path}/{i}")
        #Predict the recommended fashion
        lab, img = recommedation_model.find_style(data)
        output = {"image": i, "style": lab, "category":data}
        file.write(str(output) + "\n")