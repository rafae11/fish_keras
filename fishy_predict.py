from fishy_imports import *

#Display version
print("tensorflow version",tf.__version__)
print("keras version",keras.__version__)
 
#print(train_path)
#print(validation_path)
#print(class_one_hot)

# training image path
t_image = list()

# validation image path
v_image = list()

# training label one hot encoded
train_label = list()

# validation label one hot encoded
valid_label = list()

#generate training paths, training label, validation path, validation label
for i in range(0,len(flist)):
    append_image_paths(i,flist[i],t_image,v_image)
    append_one_hot(i,flist[i],train_label,valid_label)

#checks
#print('train samples',len(t_image))
#print('validation samples',len(v_image))
#print('train labels', len(train_label))
#print('validation labels', len(valid_label))

#show_image(t_image[0])
#show_list(t_image,train_label)
#show_list(v_image,valid_label)
#show_list(test_path,train_label)

model = inception.Inception()

#generate transfer values to be used as input for the model
transfer_values_train  = inception.transfer_values_cache(cache_path = ft_image_path,model = model,image_paths = t_image)
transfer_values_valid  = inception.transfer_values_cache(cache_path = fv_image_path,model = model,image_paths = v_image)
transfer_values_test   = inception.transfer_values_cache(cache_path = ft_test_path, model = model,image_paths = test_path)

transfer_values_test = transfer_values_test.reshape(len(transfer_values_test),32,64,1)

valid_label = np.array(valid_label)
train_label = np.array(train_label)

#print(transfer_values_train.shape)
#print(transfer_values_valid.shape)
#print(transfer_values_test.shape)
#print(train_label.shape)
#print(valid_label.shape)

batch_size = 32
nb_epoch = 10

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

model.summary()
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

score = model.predict(transfer_values_test)

target = open('fishy_submission.csv', 'w')
target.write('Image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
for i in range(0,1000):
    index = test_path[i].rfind('/')
    concats = ','.join(str(v) for v in score[i])
    target.write("%s,%s\n" %(test_path[i][index+1:],concats))
target.close()

print('classifier complete')

K.clear_session()


