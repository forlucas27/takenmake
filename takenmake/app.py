import os
from flask import Flask
from flask import flash, request, redirect, url_for, render_template
# these are my own functions that I've made
from Kmeans_watershed_segmentation import image_segmentation
from Prediction_fromPretrained import Prediction_from_Model
from Recipes_fromFoods import recipes_from_FoodList
from werkzeug.utils import secure_filename
from Link_to_Preview_Info import get_preview_info

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow

from skimage import io
import cv2
import numpy as np

UPLOAD_FOLDER = 'static/uploads' 
TEMP_FOLDER = 'static/temp' 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home_page():  # in the home page we will get the uploaded picture of the fridge interior
    global filename
    global filestr
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            print(f'filename: {filename}')

            return redirect(url_for('results'))
    return render_template('index.html')





@app.route('/results', methods=['GET', 'POST'])
def results():
    print('RESULTS')

    all_foods = image_segmentation(filename) 
    print('SEGMENTATION DONE')


    print(os.getcwd())
    for i in range(0, len(all_foods)):

        img = all_foods[i][:, :, ::-1]
        res = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        all_foods[i] = res
    all_foods1 = np.array(all_foods)

    # we also need to process the same way we processed our training data
    x_train = all_foods1.astype('float32')/255

    base = MobileNetV2(input_shape=(224, 224, 3),
                        include_top=False, weights='imagenet')

    base.trainable = True
    model = Sequential()
    model.add(base)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(47, activation='softmax'))
    print(os.getcwd())
    os.chdir(r"..")
    print(os.getcwd())
    model.load_weights(
        r'../MobileNetV2_47Classes_56000Images_78.hdf5')



    predict = model.predict(x_train)

    category_names = ['alcohol', 'apples', 'asparagus', 'bacon', 'beets', 'bell pepper', 'blackberries', 'broccoli',
                      'brussel sprouts', 'carrot', 'celery', 'chard', 'cheese', 'chicken', 'chives', 'cod', 'coleslaw mix',
                      'corn', 'cottage cheese', 'cucumber', 'deli meat', 'eggs', 'fresh herbs', 'garlic', 'grapes',
                      'green beans', 'green onion', 'hot dog', 'leafy greens', 'maple syrup', 'mushroom', 'onion', 'orange',
                      'pear', 'pesto', 'pineapple', 'pomegranate', 'raw meat', 'salmon', 'shrimp', 'soy sauce', 'strawberries',
                      'sweet potato', 'tofu', 'tomato', 'turmeric', 'yogurt']

    # we need to convert our predictions into their string label values
    zero_y = np.zeros(((predict.shape[0]), (predict.shape[1]+1)))
    argmax_lst = np.argmax(predict, axis=1)
    prob = []
    for i in range(len(argmax_lst)):

        if predict[i, argmax_lst[i]] > 0.8:
            zero_y[i][argmax_lst[i]] = 1
            prob.append(predict[i, argmax_lst[i]])
        else:
            zero_y[i][10] = 1
            prob.append(0)

    def get_classlabel(encoded, category_names):
        res = [i for i, val in enumerate(encoded) if val]
        label = str(category_names[int(res[0])])
        return label

    test_labels = []
    for i in range(0, len(predict)):
        test_labels.append(get_classlabel(zero_y[i], category_names))

    final_labels = list(set(test_labels))
    if 'None' in final_labels:
        final_labels.remove('None')


    
    print('Prediction from model done')
    titles, links, images, Tot_missedIng, Tot_usedIng = recipes_from_FoodList(
        test_labels)
    print('Recipes from food list done')
    render_template('about.html')

    return render_template('results2.html', link1=links[0], title1=titles[0], image_link1=images[0],
                           link2=links[1], title2=titles[1], image_link2=images[1],
                           link3=links[2], title3=titles[2], image_link3=images[2],
                           link4=links[3], title4=titles[3], image_link4=images[3],
                           Used1=(', '.join(Tot_usedIng[0])), Miss1=(', '.join(Tot_missedIng[0])),
                           Used2=(', '.join(Tot_usedIng[1])), Miss2=(', '.join(Tot_missedIng[1])),
                           Used3=(', '.join(Tot_usedIng[2])), Miss3=(', '.join(Tot_missedIng[2])),
                           Used4=(', '.join(Tot_usedIng[3])), Miss4=(', '.join(Tot_missedIng[3]))

                           )


@app.route('/about', methods=['GET', 'POST'])
def about():
    print('ABOUT')
    return render_template('about.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():

    return render_template('contact.html')



