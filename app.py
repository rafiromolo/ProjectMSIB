import os
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from flask import Flask, flash, redirect, render_template, request, session
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array


# ====== FLASK SETUP ======

UPLOAD_FOLDER = 'D:\\datarafi\\11 - Events\\AI Mastery Program\\Final Project\\Proyek Akhir\\static\\upload\\images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'ini secret key REKANAN'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ====== Prediction ====== 

model = load_model('model/mobileNet.h5')
classes = ['Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

def finds():
  test_datagen = ImageDataGenerator(rescale = 1./255)
  test_dir = 'D:\\datarafi\\11 - Events\\AI Mastery Program\\Final Project\\Proyek Akhir\\static\\upload'
  test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size =(224, 224),
        color_mode ="rgb",
        shuffle = False,
        class_mode = None,
        batch_size = 1)

  pred = model.predict_generator(test_generator)
  print(pred)
  
  return str(classes[np.argmax(pred)])

def predict(filename, model):
  img = load_img(filename, target_size = (224, 224))
  img = img_to_array(img)
  img = img.reshape(1, 224, 224, 3)

  img = img.astype('float32')
  img = img/255.0
  result = model.predict(img)

  print("original result:", result)

  dict_result = {}
  for i in range(10):
    # dict_result[result[0][i]] = classes[i]
    dict_result[classes[i]] = result[0][i]

  print("dict_result:", dict_result)
  sorted_prediction = sorted(dict_result.items(), key=lambda x: x[1], reverse=True)
  print("sorted_dict:", sorted_prediction)

  # res = result[0]
  # res.sort()
  # res = res[::-1]
  # prob = res[:3]

#   prob_result = []
#   class_result = []
# 
#   for i in range(3):
#     prob_result.append((prob[i]*100).round(2))
#     class_result.append(dict_result[prob[i]])

  predicted_class = sorted_prediction[0][0]
  prob_result = sorted_prediction[0][1]
  print("predicted_class:", predicted_class)
  print("prob_result:", prob_result)
  
  # return class_result, prob_result
  return predicted_class, (prob_result*100).round(2)

def food_recipe(class_result):
  print('class_result in food recipe:', class_result)
  food_df = pd.read_csv('static/food_db.csv')
  recipe_arr_index = []

  for i in range(len(food_df)):
    if food_df.loc[i, 'class'] == class_result:
      recipe_arr_index.append(i)
  
  print('isi recipe_arr_index:', recipe_arr_index)
  
  # for j in range(len(recipe_arr_index)):
  #   recipe_arr.append(food_df.loc[recipe_arr_index[j]][1].split(', '))
  recipe_1 = food_df.loc[recipe_arr_index[0]][1].split(', ')
  recipe_2 = food_df.loc[recipe_arr_index[1]][1].split(', ')
  recipe_3 = food_df.loc[recipe_arr_index[2]][1].split(', ')

    # if class_result == food_df.loc[i, 'class']:
    #   new_food_df = food_df.loc[i]
    #   recipe_1 = food_df[j]
    #   title = new_food_df[1].split(', ')
    #   recipe = new_food_df[2].split(', ')
    #   steps = new_food_df[3].split(', ')

  return recipe_1, recipe_2, recipe_3

# ====== Routes ====== 

@app.route("/")
def home():
  session['secrrt'] = 'sec'
  return render_template("index.html")

# @app.route("/about")
# def about():
#   session['secrrt'] = 'sec'
#   return render_template("about.html")

@app.route("/upload", methods=['GET', 'POST'])
def upload():
  if request.method == 'GET':
    return render_template("upload.html")
  elif request.method == 'POST':
    if 'inpFile' not in request.files:
      flash('No file part')
      return redirect(request.url)

    file = request.files['inpFile']
    print(file)

    if file.filename == '':
      flash('No selected file')
      return redirect(request.url)
    
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      img = file.filename
      print(img_path)
      print(img)
      predicted_class, prob_result = predict(img_path, model)

      print("Class Name: ", predicted_class)
      print('tipe class name:', type(predicted_class))
      print("Prob Result:", prob_result)

      # predictions = {
      #   "class1":class_result[0],
      #   "class2":class_result[1],
      #   "class3":class_result[2],
      #   "prob1": prob_result[0],
      #   "prob2": prob_result[1],
      #   "prob3": prob_result[2],
      # }

      recipe_1, recipe_2, recipe_3 = food_recipe(predicted_class)
      print('recipe_1:', recipe_1)
      print('recipe_2:', recipe_2)
      print('recipe_3:', recipe_3)
      # my_recipe_array = food_recipe(predicted_class)
      # print('recipe_arr:', my_recipe_array)
      # print('panjang recipe_arr:', len(my_recipe_array))
      # print("TIPE RECIPE:", type(recipe))

      # return render_template("upload.html", result=predictions, recipe=recipe, steps=steps, nutrition = nutrition)
      # return render_template("upload.html", result=predicted_class, result_prob=prob_result, recipe_1=recipe_1, recipe_2=recipe_2, recipe_3=recipe_3)
      return render_template("upload.html", result=predicted_class, result_prob=prob_result, recipe_1=recipe_1, recipe_2=recipe_2, recipe_3=recipe_3)
    else:
      error = 'Mohon upload gambar dengan format png, jpg, atau jpeg.'
    
  else:
    return "Unsupported Request Method"


if __name__ == '__main__':
  app.run(port=5000, debug=True)