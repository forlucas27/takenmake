def Prediction_from_Model(all_foods):

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import SGD, Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
    from tensorflow.keras.models import load_model
    import tensorflow
    import cv2
    import numpy as np
    import os

    # we need the images to have the correct shape
    for i in range(0, len(all_foods)):
        img = all_foods[i][:, :, ::-1]
        res = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        all_foods[i] = res

    all_foods = np.array(all_foods)

    # we also need to process the same way we processed our training data
    x_train = all_foods.astype('float32')/255

    # now we can get our prediction
    graph = tensorflow.compat.v1.get_default_graph()
    with graph.as_default():
        print('what')
        # create the base pre-trained model
        base = MobileNetV2(input_shape=(224, 224, 3),
                           include_top=False, weights='imagenet')
        print('hi')
        base.trainable = True
        model = Sequential()
        model.add(base)
        print('step')
        # model.add(Flatten())
        model.add(GlobalAveragePooling2D())
        # model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        print('is taking too long')
        model.add(Dense(47, activation='softmax'))
        print(os.getcwd())
        os.chdir(r"..")
        print(os.getcwd())
        model.load_weights(
            r'MobileNetV2_47Classes_56000Images_78.hdf5') #C:\Users\forlu\Documents\GitHub\Fridge_app
        print('model loaded')
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
        # print(argmax_lst[i])
        # print(predict[i,argmax_lst[i]])
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

    return final_labels
